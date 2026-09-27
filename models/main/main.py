import pickle
import math

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
from train_utils import AverageMeter

# Use a CN mirror by default for Hugging Face model/tokenizer downloads.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .main_utils import Get_Scalar
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller, MultiClassFocalLossWithAlpha
import json

from sklearn.metrics import *
from copy import deepcopy
from train_utils import ce_loss
import contextlib
from models.nets import dmd
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from mllm_evidence import MLLMEvidenceVerifier


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1)+1e-8)
        x_norm = torch.max(x_norm, 1e-8*torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1)+1e-8)
        y_norm = torch.max(y_norm, 1e-8*torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape

        s = feats.repeat(1, B).view(-1, F) # B**2 X F
        s_ids = ids.view(B, 1).repeat(1, B) # B X B
        
        t = feats.repeat(B, 1) # B**2 X F
        t_ids = ids.view(1, B).repeat(B, 1) # B X B 

        cosine = self.compute_cosine(s, t) # B**2
        equal_mask = torch.eye(B, dtype=torch.bool) # B X B
        s_ids = s_ids[~equal_mask].view(B, B-1) # B X (B-1)
        t_ids = t_ids[~equal_mask].view(B, B-1) # B X (B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B-1) # B X (B-1)

        sim_mask = (s_ids == t_ids) # B X (B-1)
        margin = 0.15 * abs(s_ids - t_ids)#[~sim_mask].view(B, B - 3)

        loss = 0
        loss_num = 0
        
        for i in range(B):
            sim_num = sum(sim_mask[i])
            dif_num = B - 1 - sim_num
            if not sim_num or not dif_num:
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1

        loss = loss / loss_num
        return loss


def soft_cross_entropy(logits, target_prob, reduction="none"):
    log_prob = F.log_softmax(logits, dim=-1)
    loss = -(target_prob * log_prob).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    return loss


def compute_ce_umc_loss(
    logits_text,
    logits_image,
    qwen_evidence,
    scope_mask=None,
    target_temperature=2.0,
    text_conf_threshold=0.90,
    image_conf_threshold=0.90,
    min_certainty=0.20,
    confidence_power=1.0,
    min_samples=2,
):
    """
    Calibrated External Unimodal Supervision.

    Detached Qwen text/image distributions softly supervise only the existing
    private classifiers. The multimodal target and fusion classifier are not
    changed. Confidence and normalized entropy jointly control each modality's
    sample weight so an overconfident but diffuse target cannot dominate UMC.
    """
    eps = 1e-8
    zero = logits_text.new_tensor(0.0)
    batch_size = logits_text.shape[0]
    if scope_mask is None:
        scope_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=logits_text.device,
        )
    else:
        scope_mask = scope_mask.detach().bool()

    def branch_loss(logits, target, valid, confidence_threshold):
        target = target.detach().float().clamp_min(0.0)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
        valid = valid.detach().bool() & scope_mask

        confidence = target.max(dim=1).values
        entropy = -(
            target * target.clamp_min(eps).log()
        ).sum(dim=1)
        max_entropy = torch.log(
            target.new_tensor(float(max(target.shape[1], 2)))
        )
        certainty = (
            1.0 - entropy / max_entropy.clamp_min(eps)
        ).clamp(0.0, 1.0)
        selected = (
            valid
            & confidence.ge(float(confidence_threshold))
            & certainty.ge(float(min_certainty))
        )
        selected_count = int(selected.sum().detach().cpu())

        stats = {
            "selected": selected,
            "selected_count": selected_count,
            "valid_ratio": float(valid.float().mean().detach().cpu()),
            "selected_ratio": float(selected.float().mean().detach().cpu()),
            "confidence_mean": 0.0,
            "certainty_mean": 0.0,
            "weight_mean": 0.0,
            "target_entropy_mean": 0.0,
            "model_evidence_agreement": 0.0,
        }
        if selected_count < int(min_samples):
            return zero, stats

        temperature = max(float(target_temperature), 1e-6)
        softened_target = F.softmax(
            target[selected].clamp_min(eps).log() / temperature,
            dim=1,
        ).detach()
        sample_weight = (
            confidence[selected].pow(float(confidence_power))
            * certainty[selected]
        ).detach()
        sample_loss = soft_cross_entropy(
            logits[selected],
            softened_target,
            reduction="none",
        )
        loss = (
            sample_weight * sample_loss
        ).sum() / sample_weight.sum().clamp_min(eps)

        softened_entropy = -(
            softened_target
            * softened_target.clamp_min(eps).log()
        ).sum(dim=1)
        stats.update({
            "confidence_mean": float(
                confidence[selected].mean().detach().cpu()
            ),
            "certainty_mean": float(
                certainty[selected].mean().detach().cpu()
            ),
            "weight_mean": float(
                sample_weight.mean().detach().cpu()
            ),
            "target_entropy_mean": float(
                softened_entropy.mean().detach().cpu()
            ),
            "model_evidence_agreement": float(
                logits[selected].detach().argmax(dim=1).eq(
                    target[selected].argmax(dim=1)
                ).float().mean().cpu()
            ),
        })
        return loss, stats

    text_loss, text_stats = branch_loss(
        logits_text,
        qwen_evidence["text_dist"],
        qwen_evidence["valid_text"],
        text_conf_threshold,
    )
    image_loss, image_stats = branch_loss(
        logits_image,
        qwen_evidence["image_dist"],
        qwen_evidence["valid_image"],
        image_conf_threshold,
    )
    stats = {
        "loss_ce_umc_text": float(text_loss.detach().cpu()),
        "loss_ce_umc_image": float(image_loss.detach().cpu()),
        "ce_umc_scope_ratio": float(
            scope_mask.float().mean().detach().cpu()
        ),
        "ce_umc_text_valid_ratio": text_stats["valid_ratio"],
        "ce_umc_image_valid_ratio": image_stats["valid_ratio"],
        "ce_umc_text_selected_ratio": text_stats["selected_ratio"],
        "ce_umc_image_selected_ratio": image_stats["selected_ratio"],
        "ce_umc_text_num_selected": text_stats["selected_count"],
        "ce_umc_image_num_selected": image_stats["selected_count"],
        "ce_umc_text_confidence_mean": text_stats["confidence_mean"],
        "ce_umc_image_confidence_mean": image_stats["confidence_mean"],
        "ce_umc_text_certainty_mean": text_stats["certainty_mean"],
        "ce_umc_image_certainty_mean": image_stats["certainty_mean"],
        "ce_umc_text_weight_mean": text_stats["weight_mean"],
        "ce_umc_image_weight_mean": image_stats["weight_mean"],
        "ce_umc_text_target_entropy_mean": text_stats[
            "target_entropy_mean"
        ],
        "ce_umc_image_target_entropy_mean": image_stats[
            "target_entropy_mean"
        ],
        "ce_umc_text_model_evidence_agreement": text_stats[
            "model_evidence_agreement"
        ],
        "ce_umc_image_model_evidence_agreement": image_stats[
            "model_evidence_agreement"
        ],
    }
    return text_loss, image_loss, stats


def compute_ucrf_loss(
    selector_lb,
    candidate_logits_lb,
    y_lb,
    selector_ulb,
    candidate_logits_ulb,
    pseudo_label,
    reliability,
    accept_mask,
    utility_temperature=0.5,
    reliability_threshold=0.75,
    min_samples=2,
    labeled_weight=1.0,
    unlabeled_weight=0.5,
):
    """
    Utility-Calibrated Residual Fusion selector supervision.

    Labeled targets are derived from each fusion candidate's true-label
    utility. Unlabeled targets use only detached, DCR-PLF accepted,
    high-reliability pseudo-label support. The targets supervise the selector;
    they never replace the SCRD pseudo-label.
    """
    eps = 1e-8
    temperature = max(float(utility_temperature), 1e-6)
    zero = candidate_logits_lb.new_tensor(0.0)

    selector_lb = selector_lb.float().clamp_min(eps)
    candidate_log_prob_lb = F.log_softmax(
        candidate_logits_lb.detach().float(),
        dim=-1,
    )
    true_label_index = y_lb.detach().long().view(-1, 1, 1).expand(-1, 3, 1)
    candidate_nll_lb = -candidate_log_prob_lb.gather(
        2,
        true_label_index,
    ).squeeze(-1)
    utility_target_lb = torch.softmax(
        -candidate_nll_lb / temperature,
        dim=1,
    ).detach()
    loss_labeled_each = (
        utility_target_lb
        * (
            utility_target_lb.clamp_min(eps).log()
            - selector_lb.log()
        )
    ).sum(dim=1)
    loss_labeled = loss_labeled_each.mean()

    reliability = reliability.detach().float().clamp(0.0, 1.0)
    selected = (
        accept_mask.detach().bool()
        & reliability.ge(float(reliability_threshold))
    )
    num_selected = int(selected.sum().detach().cpu())
    loss_unlabeled = zero
    utility_target_ulb = None
    if num_selected >= int(min_samples):
        candidate_prob_ulb = torch.softmax(
            candidate_logits_ulb[selected].detach().float(),
            dim=-1,
        )
        pseudo_index = (
            pseudo_label[selected]
            .detach()
            .long()
            .view(-1, 1, 1)
            .expand(-1, 3, 1)
        )
        candidate_support = candidate_prob_ulb.gather(
            2,
            pseudo_index,
        ).squeeze(-1)
        utility_target_ulb = torch.softmax(
            candidate_support.clamp_min(eps).log() / temperature,
            dim=1,
        ).detach()
        selector_selected = selector_ulb[selected].float().clamp_min(eps)
        loss_unlabeled_each = (
            utility_target_ulb
            * (
                utility_target_ulb.clamp_min(eps).log()
                - selector_selected.log()
            )
        ).sum(dim=1)
        selected_reliability = reliability[selected]
        selected_reliability = selected_reliability / (
            selected_reliability.mean().clamp_min(eps)
        )
        loss_unlabeled = (
            selected_reliability * loss_unlabeled_each
        ).mean()

    loss = (
        float(labeled_weight) * loss_labeled
        + float(unlabeled_weight) * loss_unlabeled
    )

    with torch.no_grad():
        candidate_pred_lb = candidate_logits_lb.argmax(dim=-1)
        candidate_correct_lb = candidate_pred_lb.eq(
            y_lb.detach().view(-1, 1)
        )
        branch_acc = candidate_correct_lb.float().mean(dim=0)
        oracle_acc = candidate_correct_lb.any(dim=1).float().mean()
        selector_choice_lb = selector_lb.argmax(dim=1)
        selector_pred_lb = candidate_pred_lb.gather(
            1,
            selector_choice_lb.unsqueeze(1),
        ).squeeze(1)
        selector_acc = selector_pred_lb.eq(y_lb.detach()).float().mean()
        utility_choice_lb = utility_target_lb.argmax(dim=1)
        selector_utility_agreement = selector_choice_lb.eq(
            utility_choice_lb
        ).float().mean()
        selector_entropy = -(
            selector_lb * selector_lb.log()
        ).sum(dim=1).mean()
        reliability_mean = (
            float(reliability[selected].mean().cpu())
            if num_selected > 0
            else 0.0
        )
        unlabeled_utility_agreement = 0.0
        if utility_target_ulb is not None:
            unlabeled_utility_agreement = float(
                selector_ulb[selected].argmax(dim=1).eq(
                    utility_target_ulb.argmax(dim=1)
                ).float().mean().cpu()
            )

    stats = {
        "loss_ucrf": float(loss.detach().cpu()),
        "loss_ucrf_labeled": float(loss_labeled.detach().cpu()),
        "loss_ucrf_unlabeled": float(loss_unlabeled.detach().cpu()),
        "ucrf_selected_ratio": float(selected.float().mean().detach().cpu()),
        "ucrf_num_selected": num_selected,
        "ucrf_reliability_mean": reliability_mean,
        "ucrf_selector_entropy": float(selector_entropy.cpu()),
        "ucrf_selector_utility_agreement": float(
            selector_utility_agreement.cpu()
        ),
        "ucrf_unlabeled_utility_agreement": unlabeled_utility_agreement,
        "ucrf_text_candidate_acc": float(branch_acc[0].cpu()),
        "ucrf_image_candidate_acc": float(branch_acc[1].cpu()),
        "ucrf_common_candidate_acc": float(branch_acc[2].cpu()),
        "ucrf_oracle_candidate_acc": float(oracle_acc.cpu()),
        "ucrf_selector_candidate_acc": float(selector_acc.cpu()),
        "ucrf_selector_mean": selector_lb.mean(dim=0).detach().cpu().tolist(),
    }
    return loss, stats, selected


def build_ec_pfd_mask(accept, reliability, threshold, neutral_uncertain=None):
    mask = accept.bool() & reliability.float().ge(float(threshold))
    if neutral_uncertain is not None:
        mask = mask & (~neutral_uncertain.bool())
    return mask


def compute_ec_pfd_loss(
    logits_common,
    logits_text_private,
    logits_image_private,
    pseudo_label,
    reliability_score,
    accept_mask,
    threshold,
    pseudo_prob=None,
    neutral_uncertain=None,
    use_soft_label=True,
    detach_target=True,
    min_samples=1,
):
    ec_pfd_mask = build_ec_pfd_mask(
        accept=accept_mask,
        reliability=reliability_score,
        threshold=threshold,
        neutral_uncertain=neutral_uncertain,
    )
    num_selected = int(ec_pfd_mask.sum().detach().cpu())
    zero = logits_common.new_tensor(0.0)
    stats = {
        "loss_ec_pfd": 0.0,
        "ec_pfd_mask_ratio": float(ec_pfd_mask.float().mean().detach().cpu()),
        "ec_pfd_num_selected": num_selected,
        "ec_pfd_reliability_mean": 0.0,
    }
    if num_selected < int(min_samples):
        return zero, stats, ec_pfd_mask

    reliability = reliability_score[ec_pfd_mask].float()
    if detach_target:
        reliability = reliability.detach()
    reliability = reliability.clamp(min=0.0)
    stats["ec_pfd_reliability_mean"] = float(reliability.mean().detach().cpu())
    reliability = reliability / (reliability.mean().detach() + 1e-6)

    if use_soft_label and pseudo_prob is not None:
        target_prob = pseudo_prob[ec_pfd_mask].float()
        if detach_target:
            target_prob = target_prob.detach()
        target_prob = target_prob / (target_prob.sum(dim=-1, keepdim=True) + 1e-6)
        loss_common = soft_cross_entropy(logits_common[ec_pfd_mask], target_prob)
        loss_text_private = soft_cross_entropy(logits_text_private[ec_pfd_mask], target_prob)
        loss_image_private = soft_cross_entropy(logits_image_private[ec_pfd_mask], target_prob)
    else:
        target = pseudo_label[ec_pfd_mask].long()
        if detach_target:
            target = target.detach()
        loss_common = F.cross_entropy(logits_common[ec_pfd_mask], target, reduction="none")
        loss_text_private = F.cross_entropy(logits_text_private[ec_pfd_mask], target, reduction="none")
        loss_image_private = F.cross_entropy(logits_image_private[ec_pfd_mask], target, reduction="none")

    loss_each = loss_common + 0.5 * (loss_text_private + loss_image_private)
    loss = (reliability * loss_each).mean()
    stats["loss_ec_pfd"] = float(loss.detach().cpu())
    return loss, stats, ec_pfd_mask


def _normalize_prob(prob):
    prob = prob.float().clamp_min(0.0)
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _js_similarity(prob_a, prob_b):
    prob_a = _normalize_prob(prob_a)
    prob_b = _normalize_prob(prob_b)
    mean_prob = 0.5 * (prob_a + prob_b)
    kl_a = (prob_a * (prob_a.clamp_min(1e-8).log() - mean_prob.clamp_min(1e-8).log())).sum(dim=-1)
    kl_b = (prob_b * (prob_b.clamp_min(1e-8).log() - mean_prob.clamp_min(1e-8).log())).sum(dim=-1)
    js = 0.5 * (kl_a + kl_b)
    return (1.0 - js / np.log(2.0)).clamp(min=0.0, max=1.0)


def _dctr_calibrated_support(raw_support, precision, num_classes):
    incorrect_mass = (1.0 - raw_support) / float(max(num_classes - 1, 1))
    return (
        precision * raw_support
        + (1.0 - precision) * incorrect_mass
    ).clamp(min=1e-4, max=1.0)


@torch.no_grad()
def compute_dctr_reliability(
    calibrator,
    weak_probs,
    strong_probs,
    evidence,
    use_aug_stability=True,
    use_js_similarity=True,
):
    """
    Estimate independent continuous reliability for multimodal/text/image.

    First calibration:
        Beta-smoothed external precision calibrates each Qwen view.
    Second calibration:
        Beta-smoothed SCRD precision, calibrated external support,
        augmentation stability, and distribution agreement are combined.
    """
    weak_probs = _normalize_prob(weak_probs.detach())
    strong_probs = _normalize_prob(strong_probs.detach())
    evidence_prob = _normalize_prob(evidence["distributions"].detach())
    evidence_valid = evidence["valid"].bool()
    relation_group = evidence["relation_group"].long()

    model_conf, model_labels = weak_probs.max(dim=-1)
    evidence_labels = evidence["labels"].long()
    external_precision = calibrator.posterior(
        labels=evidence_labels,
        relation_group=relation_group,
        source="external",
    )
    internal_precision = calibrator.posterior(
        labels=model_labels,
        relation_group=relation_group,
        source="internal",
    )

    num_classes = weak_probs.shape[-1]
    internal_support = _dctr_calibrated_support(
        raw_support=model_conf,
        precision=internal_precision,
        num_classes=num_classes,
    )
    evidence_support_raw = evidence_prob.gather(
        2,
        model_labels.unsqueeze(-1),
    ).squeeze(-1)
    external_support = _dctr_calibrated_support(
        raw_support=evidence_support_raw,
        precision=external_precision,
        num_classes=num_classes,
    )

    flat_weak = weak_probs.reshape(-1, num_classes)
    flat_strong = strong_probs.reshape(-1, num_classes)
    aug_stability = _js_similarity(flat_weak, flat_strong).view_as(model_conf)
    flat_evidence = evidence_prob.reshape(-1, num_classes)
    distribution_similarity = _js_similarity(
        flat_weak,
        flat_evidence,
    ).view_as(model_conf)

    if not use_aug_stability:
        aug_stability = torch.ones_like(model_conf)
    if not use_js_similarity:
        distribution_similarity = torch.ones_like(model_conf)

    # Missing evidence falls back to internally calibrated confidence instead
    # of rejecting the sample or inventing external support.
    external_support = torch.where(
        evidence_valid,
        external_support,
        internal_support,
    )
    distribution_similarity = torch.where(
        evidence_valid,
        distribution_similarity,
        torch.ones_like(distribution_similarity),
    )

    log_reliability = (
        internal_support.clamp_min(1e-4).log()
        + external_support.clamp_min(1e-4).log()
        + 0.5 * aug_stability.clamp_min(1e-4).log()
        + 0.5 * distribution_similarity.clamp_min(1e-4).log()
    ) / 3.0
    reliability = log_reliability.exp().clamp(min=0.0, max=1.0)

    return reliability.detach(), {
        "model_labels": model_labels.detach(),
        "external_precision": external_precision.detach(),
        "internal_precision": internal_precision.detach(),
        "internal_support": internal_support.detach(),
        "external_support": external_support.detach(),
        "aug_stability": aug_stability.detach(),
        "distribution_similarity": distribution_similarity.detach(),
    }


def build_msd_evidence_prior(
    text_prob,
    image_prob,
    multimodal_prob,
    q_text,
    q_image,
    q_multimodal,
    temperature=0.5,
    smoothing=0.05,
    prior_mode="evidence",
):
    """
    Build modality support in text, image, multimodal order.

    Evidence confidence alone is not treated as modality reliability. Each
    support score also requires agreement between the detached SCRD branch
    distribution and its corresponding detached Qwen distribution.
    """
    model_probs = [
        _normalize_prob(text_prob.detach()),
        _normalize_prob(image_prob.detach()),
        _normalize_prob(multimodal_prob.detach()),
    ]
    qwen_probs = [
        _normalize_prob(q_text.detach()),
        _normalize_prob(q_image.detach()),
        _normalize_prob(q_multimodal.detach()),
    ]
    if prior_mode == "uniform":
        prior = torch.full(
            (text_prob.shape[0], 3),
            1.0 / 3.0,
            dtype=text_prob.dtype,
            device=text_prob.device,
        )
    elif prior_mode == "random":
        prior = torch.rand(
            text_prob.shape[0],
            3,
            dtype=text_prob.dtype,
            device=text_prob.device,
        )
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-6)
    elif prior_mode == "evidence":
        support_scores = []
        for model_prob, qwen_prob in zip(model_probs, qwen_probs):
            qwen_confidence = qwen_prob.max(dim=1).values
            support_scores.append(qwen_confidence * _js_similarity(model_prob, qwen_prob))
        support_scores = torch.stack(support_scores, dim=1)
        prior = F.softmax(support_scores / max(float(temperature), 1e-6), dim=1)
    else:
        raise ValueError(f"Unknown MSD prior mode: {prior_mode}")

    smoothing = min(max(float(smoothing), 0.0), 1.0)
    if smoothing > 0.0:
        prior = (1.0 - smoothing) * prior + smoothing / 3.0
    return prior.detach()


def build_msd_labeled_prior(
    logits_text,
    logits_image,
    logits_multimodal,
    target,
    temperature=0.5,
    smoothing=0.05,
):
    branch_losses = torch.stack(
        [
            F.cross_entropy(logits_text.detach(), target, reduction="none"),
            F.cross_entropy(logits_image.detach(), target, reduction="none"),
            F.cross_entropy(logits_multimodal.detach(), target, reduction="none"),
        ],
        dim=1,
    )
    prior = F.softmax(-branch_losses / max(float(temperature), 1e-6), dim=1)
    smoothing = min(max(float(smoothing), 0.0), 1.0)
    if smoothing > 0.0:
        prior = (1.0 - smoothing) * prior + smoothing / 3.0
    return prior.detach()


def _distribution_kl(target_prob, predicted_prob):
    target_prob = _normalize_prob(target_prob)
    predicted_prob = _normalize_prob(predicted_prob)
    return (
        target_prob
        * (target_prob.clamp_min(1e-8).log() - predicted_prob.clamp_min(1e-8).log())
    ).sum(dim=1)


def _entropy_confidence(prob):
    prob = _normalize_prob(prob)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1)
    max_entropy = np.log(max(int(prob.shape[1]), 2))
    return (1.0 - entropy / max_entropy).clamp(min=0.0, max=1.0)


def _weighted_mean(values, weights):
    # Normalize by selected samples rather than weight sum so confidence and
    # neutral scaling change the actual feedback strength, not just ranking.
    return (values * weights).sum() / max(int(values.numel()), 1)


def compute_ead_a_loss(
    logits_text_recon,
    logits_image_recon,
    common_text,
    common_image,
    qwen_evidence,
    reliability,
    accept_mask,
    reliability_threshold=0.45,
    min_samples=2,
    common_align_weight=0.2,
    conflict_boost=0.5,
    neutral_scale=0.5,
    opposite_common_scale=0.0,
):
    """
    Evidence-Adaptive Disentanglement Alignment.

    Reliable EC-PLF samples receive modality-specific semantic reconstruction
    targets. Text-image evidence agreement strengthens common alignment, while
    disagreement preserves modality-private semantics instead of forcing the
    two common features together.
    """
    zero = logits_text_recon.new_tensor(0.0)
    valid = qwen_evidence["valid_text"].bool() & qwen_evidence["valid_image"].bool()
    selected = (
        accept_mask.bool()
        & valid
        & reliability.float().ge(float(reliability_threshold))
    )
    num_selected = int(selected.sum().detach().cpu())
    stats = {
        "loss_ead_a": 0.0,
        "ead_a_text_anchor_loss": 0.0,
        "ead_a_image_anchor_loss": 0.0,
        "ead_a_common_align_loss": 0.0,
        "ead_a_selected_ratio": float(selected.float().mean().detach().cpu()),
        "ead_a_num_selected": num_selected,
        "ead_a_reliability_mean": 0.0,
        "ead_a_text_conf_mean": 0.0,
        "ead_a_image_conf_mean": 0.0,
        "ead_a_agreement_mean": 0.0,
        "ead_a_opposite_ratio": 0.0,
        "ead_a_neutral_ratio": 0.0,
        "ead_a_text_weight_mean": 0.0,
        "ead_a_image_weight_mean": 0.0,
        "ead_a_common_weight_mean": 0.0,
    }
    if num_selected < int(min_samples):
        return zero, stats, selected

    q_text = _normalize_prob(qwen_evidence["text_dist"][selected].detach())
    q_image = _normalize_prob(qwen_evidence["image_dist"][selected].detach())
    trust = reliability[selected].float().detach().clamp(min=0.0, max=1.0)
    text_conf = _entropy_confidence(q_text).detach()
    image_conf = _entropy_confidence(q_image).detach()
    agreement = _js_similarity(q_text, q_image).detach()
    opposite = qwen_evidence["opposite_conflict"][selected].bool().detach()
    neutral = qwen_evidence["neutral_consensus"][selected].bool().detach()

    disagreement_boost = 1.0 + float(conflict_boost) * (1.0 - agreement)
    neutral_factor = torch.where(
        neutral,
        torch.full_like(trust, float(neutral_scale)),
        torch.ones_like(trust),
    )
    text_weight = trust * text_conf * disagreement_boost * neutral_factor
    image_weight = trust * image_conf * disagreement_boost * neutral_factor

    common_weight = trust * torch.minimum(text_conf, image_conf) * agreement
    common_weight = common_weight * neutral_factor
    common_weight = torch.where(
        opposite,
        common_weight * float(opposite_common_scale),
        common_weight,
    )

    text_kl = F.kl_div(
        F.log_softmax(logits_text_recon[selected], dim=1),
        q_text,
        reduction="none",
    ).sum(dim=1)
    image_kl = F.kl_div(
        F.log_softmax(logits_image_recon[selected], dim=1),
        q_image,
        reduction="none",
    ).sum(dim=1)
    common_distance = 1.0 - F.cosine_similarity(
        common_text[selected],
        common_image[selected],
        dim=1,
    )

    loss_text = _weighted_mean(text_kl, text_weight)
    loss_image = _weighted_mean(image_kl, image_weight)
    loss_common = _weighted_mean(common_distance, common_weight)
    loss = loss_text + loss_image + float(common_align_weight) * loss_common

    stats.update({
        "loss_ead_a": float(loss.detach().cpu()),
        "ead_a_text_anchor_loss": float(loss_text.detach().cpu()),
        "ead_a_image_anchor_loss": float(loss_image.detach().cpu()),
        "ead_a_common_align_loss": float(loss_common.detach().cpu()),
        "ead_a_reliability_mean": float(trust.mean().detach().cpu()),
        "ead_a_text_conf_mean": float(text_conf.mean().detach().cpu()),
        "ead_a_image_conf_mean": float(image_conf.mean().detach().cpu()),
        "ead_a_agreement_mean": float(agreement.mean().detach().cpu()),
        "ead_a_opposite_ratio": float(opposite.float().mean().detach().cpu()),
        "ead_a_neutral_ratio": float(neutral.float().mean().detach().cpu()),
        "ead_a_text_weight_mean": float(text_weight.mean().detach().cpu()),
        "ead_a_image_weight_mean": float(image_weight.mean().detach().cpu()),
        "ead_a_common_weight_mean": float(common_weight.mean().detach().cpu()),
    })
    return loss, stats, selected


def compute_sa_dd_loss(
    common_text,
    common_image,
    private_text,
    private_image,
    weak_prob,
    strong_prob,
    qwen_evidence,
    history_stability,
    pseudo_label,
    ecplf_accept,
    ecplf_weight,
    num_classes,
    min_history_stability=0.6,
    min_aug_stability=0.6,
    min_relation_conf=0.2,
    relation_floor=0.2,
    relation_ceiling=0.8,
    common_weight=1.0,
    private_weight=1.0,
    ecplf_floor=0.5,
    neutral_scale=0.5,
    min_samples=2,
    collapse_backoff=False,
    version="v2",
    positive_id=0,
    negative_id=1,
    v2_min_gate=0.05,
    v2_stability_temperature=0.1,
    ablation="none",
    fixed_gate_value=0.5,
):
    """
    Stability-aware relation-adaptive dynamic disentanglement regularization.

    V1 retains the original hard selection and relation interpolation for
    reproducibility. V2 applies a local residual correction: relation evidence
    decides whether common or private features are regularized, while model
    uncertainty, EC-PLF risk, and temporal/augmentation stability determine the
    intervention strength. The evidence never supplies a classification target.
    """
    version = str(version).lower()
    if version not in {"v1", "v2"}:
        raise ValueError("SA-DD version must be either 'v1' or 'v2'.")
    ablation = str(ablation).lower()
    if ablation not in {
        "none",
        "wo_relation",
        "wo_stability",
        "fixed_gate",
        "unified_reliability",
    }:
        raise ValueError(
            "SA-DD ablation must be one of: none, wo_relation, "
            "wo_stability, fixed_gate, unified_reliability."
        )
    if version != "v2" and ablation == "unified_reliability":
        raise ValueError("unified_reliability ablation is defined for SA-DD/RAD v2.")
    if not 0 <= int(positive_id) < int(num_classes):
        raise ValueError("positive_id is outside the configured class range.")
    if not 0 <= int(negative_id) < int(num_classes):
        raise ValueError("negative_id is outside the configured class range.")
    if int(positive_id) == int(negative_id):
        raise ValueError("positive_id and negative_id must be different.")

    zero = common_text.new_tensor(0.0)
    batch_size = int(common_text.shape[0])
    stats = {
        "loss_sa_dd": 0.0,
        "sa_dd_common_loss": 0.0,
        "sa_dd_private_loss": 0.0,
        "sa_dd_selected_ratio": 0.0,
        "sa_dd_num_selected": 0,
        "sa_dd_gate_mean": 0.0,
        "sa_dd_history_stability_mean": 0.0,
        "sa_dd_aug_stability_mean": 0.0,
        "sa_dd_relation_agreement_mean": 0.0,
        "sa_dd_relation_conf_mean": 0.0,
        "sa_dd_stability_gate_mean": 0.0,
        "sa_dd_model_uncertainty_mean": 0.0,
        "sa_dd_ecplf_risk_mean": 0.0,
        "sa_dd_information_mean": 0.0,
        "sa_dd_relation_strength_mean": 0.0,
        "sa_dd_same_score_mean": 0.0,
        "sa_dd_opposite_score_mean": 0.0,
        "sa_dd_uncertain_score_mean": 0.0,
        "sa_dd_common_ratio_mean": 0.0,
        "sa_dd_private_ratio_mean": 0.0,
        "sa_dd_selected_accept_ratio": 0.0,
        "sa_dd_selected_reject_ratio": 0.0,
        "sa_dd_opposite_ratio": 0.0,
        "sa_dd_neutral_ratio": 0.0,
        "sa_dd_collapse_backoff": float(bool(collapse_backoff)),
        "sa_dd_class_weight_mass": [0.0 for _ in range(int(num_classes))],
    }
    empty_mask = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=common_text.device,
    )
    if collapse_backoff or batch_size == 0:
        return zero, stats, empty_mask

    valid = qwen_evidence["valid_text"].bool() & qwen_evidence["valid_image"].bool()
    q_text = _normalize_prob(qwen_evidence["text_dist"].detach())
    q_image = _normalize_prob(qwen_evidence["image_dist"].detach())
    relation_agreement = _js_similarity(q_text, q_image).detach()
    relation_conf = torch.minimum(
        _entropy_confidence(q_text),
        _entropy_confidence(q_image),
    ).detach()
    opposite = qwen_evidence["opposite_conflict"].bool().detach()
    neutral = qwen_evidence["neutral_consensus"].bool().detach()
    relation_agreement = torch.where(
        opposite,
        torch.zeros_like(relation_agreement),
        relation_agreement,
    )

    aug_stability = _js_similarity(
        _normalize_prob(weak_prob.detach()),
        _normalize_prob(strong_prob.detach()),
    ).detach()
    history_stability = history_stability.detach().float().clamp(0.0, 1.0)
    weak_prob_normalized = _normalize_prob(weak_prob.detach())
    model_uncertainty = (
        -(weak_prob_normalized * weak_prob_normalized.clamp_min(1e-8).log()).sum(dim=1)
        / math.log(max(int(num_classes), 2))
    ).clamp(0.0, 1.0)
    ecplf_weight = ecplf_weight.detach().float().clamp(0.0, 1.0)
    ecplf_accept = ecplf_accept.detach().bool()
    ecplf_risk = torch.where(
        ecplf_accept,
        1.0 - ecplf_weight,
        torch.ones_like(ecplf_weight),
    )

    same_score = (q_text * q_image).sum(dim=1).clamp(0.0, 1.0)
    opposite_score = (
        q_text[:, int(positive_id)] * q_image[:, int(negative_id)]
        + q_text[:, int(negative_id)] * q_image[:, int(positive_id)]
    ).clamp(0.0, 1.0)
    uncertain_score = (1.0 - same_score - opposite_score).clamp(0.0, 1.0)
    relation_strength = (
        relation_conf * (same_score + opposite_score).clamp(0.0, 1.0)
    ).detach()

    neutral_factor = torch.where(
        neutral,
        torch.full_like(relation_conf, float(neutral_scale)),
        torch.ones_like(relation_conf),
    )
    if ablation == "wo_relation":
        valid = torch.ones_like(valid, dtype=torch.bool)
        relation_agreement = torch.full_like(relation_agreement, 0.5)
        relation_conf = torch.ones_like(relation_conf)
        opposite = torch.zeros_like(opposite, dtype=torch.bool)
        neutral = torch.zeros_like(neutral, dtype=torch.bool)
        same_score = torch.full_like(same_score, 0.5)
        opposite_score = torch.full_like(opposite_score, 0.5)
        uncertain_score = torch.zeros_like(uncertain_score)
        relation_strength = torch.ones_like(relation_strength)
        neutral_factor = torch.ones_like(neutral_factor)

    if version == "v1":
        selected = (
            valid
            & history_stability.ge(float(min_history_stability))
            & aug_stability.ge(float(min_aug_stability))
            & relation_conf.ge(float(min_relation_conf))
        )
        floor = float(ecplf_floor)
        accepted_trust = floor + (1.0 - floor) * ecplf_weight
        plf_relation_trust = torch.where(
            ecplf_accept,
            accepted_trust,
            torch.full_like(accepted_trust, floor),
        )
        stability_gate = history_stability * aug_stability
        information = torch.zeros_like(relation_conf)
        gate = (
            stability_gate
            * relation_conf
            * plf_relation_trust
            * neutral_factor
        ).detach()
        low = min(float(relation_floor), float(relation_ceiling))
        high = max(float(relation_floor), float(relation_ceiling))
        common_ratio = (
            float(relation_floor)
            + (float(relation_ceiling) - float(relation_floor))
            * relation_agreement
        ).clamp(min=low, max=high)
        private_ratio = 1.0 - common_ratio
    else:
        temperature = max(float(v2_stability_temperature), 1e-6)
        history_gate = torch.sigmoid(
            (history_stability - float(min_history_stability)) / temperature
        )
        augmentation_gate = torch.sigmoid(
            (aug_stability - float(min_aug_stability)) / temperature
        )
        if ablation in {"wo_stability", "fixed_gate", "unified_reliability"}:
            stability_gate = torch.ones_like(history_gate)
        else:
            stability_gate = history_gate * augmentation_gate
        if ablation == "wo_relation":
            information = (model_uncertainty + ecplf_risk) / 2.0
            common_ratio = torch.full_like(same_score, 0.5)
            private_ratio = torch.full_like(opposite_score, 0.5)
        else:
            information = (
                model_uncertainty + ecplf_risk + opposite_score
            ) / 3.0
            common_ratio = same_score
            private_ratio = opposite_score
        if ablation == "unified_reliability":
            gate = (
                ecplf_accept.float() * ecplf_weight.float()
            ).clamp(0.0, 1.0).detach()
            information = gate
            selected = valid & gate.gt(0.0)
        elif ablation == "fixed_gate":
            gate = torch.full_like(
                relation_conf,
                float(fixed_gate_value),
            ).clamp(0.0, 1.0).detach()
            selected = valid & relation_conf.ge(float(min_relation_conf))
        else:
            gate = (
                stability_gate
                * relation_strength
                * information
                * neutral_factor
            ).detach()
            selected = (
                valid
                & relation_conf.ge(float(min_relation_conf))
                & gate.ge(max(float(v2_min_gate), 0.0))
            )

    num_selected = int(selected.sum().detach().cpu())
    stats["sa_dd_selected_ratio"] = float(selected.float().mean().detach().cpu())
    stats["sa_dd_num_selected"] = num_selected
    if num_selected < int(min_samples):
        return zero, stats, selected

    common_distance = 1.0 - F.cosine_similarity(
        common_text,
        common_image,
        dim=1,
    )
    private_overlap = F.cosine_similarity(
        private_text,
        private_image,
        dim=1,
    ).abs()
    common_each = float(common_weight) * common_ratio * common_distance
    private_each = float(private_weight) * private_ratio * private_overlap

    class_losses = []
    class_common = []
    class_private = []
    class_weight_mass = []
    for class_id in range(int(num_classes)):
        class_mask = selected & pseudo_label.eq(class_id)
        class_gate = gate[class_mask]
        mass = float(class_gate.sum().detach().cpu()) if class_gate.numel() else 0.0
        class_weight_mass.append(mass)
        if class_gate.numel() == 0 or float(class_gate.sum().detach().cpu()) <= 0.0:
            continue
        denom = class_gate.sum() + 1e-6
        class_common_loss = (class_gate * common_each[class_mask]).sum() / denom
        class_private_loss = (class_gate * private_each[class_mask]).sum() / denom
        class_common.append(class_common_loss)
        class_private.append(class_private_loss)
        class_losses.append(class_common_loss + class_private_loss)

    if not class_losses:
        return zero, stats, selected

    loss_common = torch.stack(class_common).mean()
    loss_private = torch.stack(class_private).mean()
    loss = torch.stack(class_losses).mean()
    selected_gate = gate[selected]
    selected_accept = ecplf_accept[selected].float()
    stats.update({
        "loss_sa_dd": float(loss.detach().cpu()),
        "sa_dd_common_loss": float(loss_common.detach().cpu()),
        "sa_dd_private_loss": float(loss_private.detach().cpu()),
        "sa_dd_gate_mean": float(selected_gate.mean().detach().cpu()),
        "sa_dd_history_stability_mean": float(history_stability[selected].mean().detach().cpu()),
        "sa_dd_aug_stability_mean": float(aug_stability[selected].mean().detach().cpu()),
        "sa_dd_relation_agreement_mean": float(relation_agreement[selected].mean().detach().cpu()),
        "sa_dd_relation_conf_mean": float(relation_conf[selected].mean().detach().cpu()),
        "sa_dd_stability_gate_mean": float(stability_gate[selected].mean().detach().cpu()),
        "sa_dd_model_uncertainty_mean": float(model_uncertainty[selected].mean().detach().cpu()),
        "sa_dd_ecplf_risk_mean": float(ecplf_risk[selected].mean().detach().cpu()),
        "sa_dd_information_mean": float(information[selected].mean().detach().cpu()),
        "sa_dd_relation_strength_mean": float(relation_strength[selected].mean().detach().cpu()),
        "sa_dd_same_score_mean": float(same_score[selected].mean().detach().cpu()),
        "sa_dd_opposite_score_mean": float(opposite_score[selected].mean().detach().cpu()),
        "sa_dd_uncertain_score_mean": float(uncertain_score[selected].mean().detach().cpu()),
        "sa_dd_common_ratio_mean": float(common_ratio[selected].mean().detach().cpu()),
        "sa_dd_private_ratio_mean": float(private_ratio[selected].mean().detach().cpu()),
        "sa_dd_selected_accept_ratio": float(selected_accept.mean().detach().cpu()),
        "sa_dd_selected_reject_ratio": float((1.0 - selected_accept).mean().detach().cpu()),
        "sa_dd_opposite_ratio": float(opposite[selected].float().mean().detach().cpu()),
        "sa_dd_neutral_ratio": float(neutral[selected].float().mean().detach().cpu()),
        "sa_dd_class_weight_mass": class_weight_mass,
    })
    return loss, stats, selected


def compute_msd_loss(
    selector_lb,
    logits_text_lb,
    logits_image_lb,
    logits_multimodal_lb,
    y_lb,
    selector_ulb,
    text_prob_ulb,
    image_prob_ulb,
    multimodal_prob_ulb,
    qwen_evidence,
    reliability,
    accept_mask,
    reliability_threshold=0.75,
    support_temperature=0.5,
    labeled_temperature=0.5,
    smoothing=0.05,
    labeled_weight=1.0,
    unlabeled_weight=1.0,
    use_labeled_anchor=True,
    use_unlabeled_evidence=True,
    exclude_opposite_conflict=True,
    exclude_neutral_consensus=False,
    prior_mode="evidence",
    y_ulb=None,
):
    zero = selector_lb.new_tensor(0.0)
    loss_labeled = zero
    loss_unlabeled = zero
    labeled_prior = None
    evidence_prior = None

    if use_labeled_anchor and selector_lb.numel() > 0:
        labeled_prior = build_msd_labeled_prior(
            logits_text_lb,
            logits_image_lb,
            logits_multimodal_lb,
            y_lb,
            temperature=labeled_temperature,
            smoothing=smoothing,
        )
        loss_labeled = _distribution_kl(labeled_prior, selector_lb).mean()

    valid_all = qwen_evidence["valid_all"].bool()
    selected = (
        valid_all
        & accept_mask.bool()
        & reliability.float().ge(float(reliability_threshold))
    )
    if exclude_opposite_conflict:
        selected = selected & (~qwen_evidence["opposite_conflict"].bool())
    if exclude_neutral_consensus:
        selected = selected & (~qwen_evidence["neutral_consensus"].bool())

    selected_count = int(selected.sum().detach().cpu())
    selected_weight_mean = 0.0
    selected_prior_correct = 0.0
    text_branch_acc = 0.0
    image_branch_acc = 0.0
    multimodal_branch_acc = 0.0
    oracle_branch_acc = 0.0
    selector_branch_acc = 0.0
    branch_predictions = torch.stack(
        [
            text_prob_ulb.argmax(dim=1),
            image_prob_ulb.argmax(dim=1),
            multimodal_prob_ulb.argmax(dim=1),
        ],
        dim=1,
    )
    if y_ulb is not None:
        branch_correct = branch_predictions.eq(y_ulb.view(-1, 1))
        text_branch_acc = float(branch_correct[:, 0].float().mean().detach().cpu())
        image_branch_acc = float(branch_correct[:, 1].float().mean().detach().cpu())
        multimodal_branch_acc = float(branch_correct[:, 2].float().mean().detach().cpu())
        oracle_branch_acc = float(branch_correct.any(dim=1).float().mean().detach().cpu())
        selector_choice = selector_ulb.detach().argmax(dim=1)
        selector_prediction = branch_predictions.gather(1, selector_choice.view(-1, 1)).squeeze(1)
        selector_branch_acc = float(selector_prediction.eq(y_ulb).float().mean().detach().cpu())

    if use_unlabeled_evidence and selected_count > 0:
        evidence_prior = build_msd_evidence_prior(
            text_prob_ulb,
            image_prob_ulb,
            multimodal_prob_ulb,
            qwen_evidence["text_dist"],
            qwen_evidence["image_dist"],
            qwen_evidence["multimodal_dist"],
            temperature=support_temperature,
            smoothing=smoothing,
            prior_mode=prior_mode,
        )
        sample_loss = _distribution_kl(evidence_prior[selected], selector_ulb[selected])
        sample_weight = reliability[selected].float().detach().clamp_min(0.0)
        loss_unlabeled = (sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        selected_weight_mean = float(sample_weight.mean().detach().cpu())

        if y_ulb is not None:
            prior_choice = evidence_prior.argmax(dim=1)
            chosen_prediction = branch_predictions.gather(1, prior_choice.view(-1, 1)).squeeze(1)
            selected_prior_correct = float(
                chosen_prediction[selected].eq(y_ulb[selected]).float().mean().detach().cpu()
            )

    loss = float(labeled_weight) * loss_labeled + float(unlabeled_weight) * loss_unlabeled
    selector_mean = selector_ulb.detach().mean(dim=0) if selector_ulb.numel() else selector_lb.new_zeros(3)
    if evidence_prior is not None and selected_count > 0:
        prior_mean = evidence_prior[selected].mean(dim=0)
    else:
        prior_mean = selector_lb.new_zeros(3)
    stats = {
        "loss_msd": float(loss.detach().cpu()),
        "loss_msd_labeled": float(loss_labeled.detach().cpu()),
        "loss_msd_unlabeled": float(loss_unlabeled.detach().cpu()),
        "msd_selected_ratio": float(selected.float().mean().detach().cpu()),
        "msd_num_selected": selected_count,
        "msd_reliability_mean": selected_weight_mean,
        "msd_valid_evidence_ratio": float(valid_all.float().mean().detach().cpu()),
        "msd_opposite_conflict_ratio": float(
            qwen_evidence["opposite_conflict"].float().mean().detach().cpu()
        ),
        "msd_neutral_consensus_ratio": float(
            qwen_evidence["neutral_consensus"].float().mean().detach().cpu()
        ),
        "msd_text_branch_acc": text_branch_acc,
        "msd_image_branch_acc": image_branch_acc,
        "msd_multimodal_branch_acc": multimodal_branch_acc,
        "msd_oracle_branch_acc": oracle_branch_acc,
        "msd_selector_branch_acc": selector_branch_acc,
        "msd_prior_selected_branch_acc": selected_prior_correct,
        "msd_selector_mean": selector_mean.cpu().tolist(),
        "msd_prior_mean": prior_mean.detach().cpu().tolist(),
    }
    return loss, stats, selected


def build_dctr_msg_prior(
    dctr_reliability,
    temperature=0.5,
    smoothing=0.05,
):
    """
    Convert DCTR reliability from [multimodal, text, image] into the selector
    order [text, image, multimodal].

    The logarithm makes the prior depend on relative reliability instead of
    treating the bounded reliability values as unconstrained selector logits.
    """
    selector_reliability = dctr_reliability.detach()[:, [1, 2, 0]]
    selector_reliability = selector_reliability.clamp_min(1e-6)
    prior = F.softmax(
        selector_reliability.log() / max(float(temperature), 1e-6),
        dim=1,
    )
    smoothing = min(max(float(smoothing), 0.0), 1.0)
    if smoothing > 0.0:
        prior = (1.0 - smoothing) * prior + smoothing / prior.shape[1]
    return prior.detach()


def compute_dctr_msg_loss(
    selector_lb,
    logits_text_lb,
    logits_image_lb,
    logits_multimodal_lb,
    y_lb,
    selector_ulb,
    text_prob_ulb,
    image_prob_ulb,
    multimodal_prob_ulb,
    dctr_reliability,
    accept_mask,
    opposite_conflict=None,
    reliability_threshold=0.45,
    prior_temperature=0.5,
    labeled_temperature=0.5,
    smoothing=0.05,
    min_margin=0.02,
    min_samples=2,
    labeled_weight=1.0,
    unlabeled_weight=1.0,
    use_labeled_anchor=True,
    exclude_opposite_conflict=False,
    y_ulb=None,
):
    """
    DCTR-MSG distills calibrated branch reliability into SCRD's soft selector.

    DCTR reliability is used only as a detached training target. Inference
    always uses the learned selector and therefore does not require evidence.
    """
    zero = selector_lb.new_tensor(0.0)
    loss_labeled = zero
    loss_unlabeled = zero

    if use_labeled_anchor and selector_lb.numel() > 0:
        labeled_prior = build_msd_labeled_prior(
            logits_text_lb,
            logits_image_lb,
            logits_multimodal_lb,
            y_lb,
            temperature=labeled_temperature,
            smoothing=smoothing,
        )
        loss_labeled = _distribution_kl(
            labeled_prior,
            selector_lb,
        ).mean()

    reliability = dctr_reliability.detach().float().clamp(0.0, 1.0)
    reliability_prior = build_dctr_msg_prior(
        reliability,
        temperature=prior_temperature,
        smoothing=smoothing,
    )
    reliability_margin = (
        reliability.max(dim=1).values - reliability.min(dim=1).values
    )
    selected = (
        accept_mask.bool()
        & reliability[:, 0].ge(float(reliability_threshold))
        & reliability_margin.ge(float(min_margin))
    )
    if exclude_opposite_conflict and opposite_conflict is not None:
        selected = selected & (~opposite_conflict.bool())

    selected_count = int(selected.sum().detach().cpu())
    if selected_count < int(min_samples):
        selected = torch.zeros_like(selected)
        selected_count = 0
    else:
        sample_loss = _distribution_kl(
            reliability_prior[selected],
            selector_ulb[selected],
        )
        sample_weight = reliability[selected, 0].clamp_min(0.0)
        loss_unlabeled = (
            sample_loss * sample_weight
        ).sum() / sample_weight.sum().clamp_min(1e-6)

    selector_mean = selector_lb.new_zeros(3)
    prior_mean = selector_lb.new_zeros(3)
    selector_entropy = 0.0
    prior_entropy = 0.0
    reliability_mean = 0.0
    reliability_margin_mean = 0.0
    selector_prior_agreement = 0.0
    selector_branch_acc = 0.0
    prior_branch_acc = 0.0
    oracle_branch_acc = 0.0
    branch_acc = [0.0, 0.0, 0.0]

    if selected_count > 0:
        selected_selector = selector_ulb[selected]
        selected_prior = reliability_prior[selected]
        selector_mean = selected_selector.detach().mean(dim=0)
        prior_mean = selected_prior.mean(dim=0)
        selector_entropy = float(
            (
                -selected_selector.detach()
                * selected_selector.detach().clamp_min(1e-7).log()
            ).sum(dim=1).mean().cpu()
        )
        prior_entropy = float(
            (
                -selected_prior * selected_prior.clamp_min(1e-7).log()
            ).sum(dim=1).mean().cpu()
        )
        reliability_mean = float(
            reliability[selected, 0].mean().cpu()
        )
        reliability_margin_mean = float(
            reliability_margin[selected].mean().cpu()
        )
        selector_choice = selected_selector.detach().argmax(dim=1)
        prior_choice = selected_prior.argmax(dim=1)
        selector_prior_agreement = float(
            selector_choice.eq(prior_choice).float().mean().cpu()
        )

        if y_ulb is not None:
            branch_predictions = torch.stack(
                [
                    text_prob_ulb.argmax(dim=1),
                    image_prob_ulb.argmax(dim=1),
                    multimodal_prob_ulb.argmax(dim=1),
                ],
                dim=1,
            )
            selected_predictions = branch_predictions[selected]
            selected_targets = y_ulb[selected].view(-1, 1)
            branch_correct = selected_predictions.eq(selected_targets)
            branch_acc = (
                branch_correct.float().mean(dim=0).detach().cpu().tolist()
            )
            oracle_branch_acc = float(
                branch_correct.any(dim=1).float().mean().cpu()
            )
            selector_prediction = selected_predictions.gather(
                1,
                selector_choice.view(-1, 1),
            ).squeeze(1)
            prior_prediction = selected_predictions.gather(
                1,
                prior_choice.view(-1, 1),
            ).squeeze(1)
            flat_targets = selected_targets.squeeze(1)
            selector_branch_acc = float(
                selector_prediction.eq(flat_targets).float().mean().cpu()
            )
            prior_branch_acc = float(
                prior_prediction.eq(flat_targets).float().mean().cpu()
            )

    loss = (
        float(labeled_weight) * loss_labeled
        + float(unlabeled_weight) * loss_unlabeled
    )
    stats = {
        "loss_dctr_msg": float(loss.detach().cpu()),
        "loss_dctr_msg_labeled": float(loss_labeled.detach().cpu()),
        "loss_dctr_msg_unlabeled": float(loss_unlabeled.detach().cpu()),
        "dctr_msg_selected_ratio": float(selected.float().mean().detach().cpu()),
        "dctr_msg_num_selected": selected_count,
        "dctr_msg_reliability_mean": reliability_mean,
        "dctr_msg_reliability_margin_mean": reliability_margin_mean,
        "dctr_msg_selector_entropy": selector_entropy,
        "dctr_msg_prior_entropy": prior_entropy,
        "dctr_msg_selector_prior_agreement": selector_prior_agreement,
        "dctr_msg_text_branch_acc": float(branch_acc[0]),
        "dctr_msg_image_branch_acc": float(branch_acc[1]),
        "dctr_msg_multimodal_branch_acc": float(branch_acc[2]),
        "dctr_msg_oracle_branch_acc": oracle_branch_acc,
        "dctr_msg_selector_branch_acc": selector_branch_acc,
        "dctr_msg_prior_branch_acc": prior_branch_acc,
        "dctr_msg_selector_mean": selector_mean.detach().cpu().tolist(),
        "dctr_msg_prior_mean": prior_mean.detach().cpu().tolist(),
    }
    return loss, stats, selected


class S2_VER:
    # def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
    #              hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, tb_log=None, args=None, logger=None):

        super(S2_VER, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        # self.model = net_builder(num_classes=num_classes)
        # self.fusion_model = fusion_model.FusionModel(num_classes=num_classes)
        self.model = dmd.DMD(args)
        self.ema_model = None

        # self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.use_mllm_verification = bool(
            getattr(args, "use_mllm_verification", False)
        )
        risk_only_actions = {"risk_veto", "risk_soft_weight"}
        mllm_action = getattr(args, "mllm_action", "veto")
        evidence_needed = (
            (self.use_mllm_verification and mllm_action not in risk_only_actions)
            or bool(getattr(args, "use_msd", False))
            or bool(getattr(args, "use_ce_umc", False))
            or bool(getattr(args, "use_ec_pfd", False))
            or bool(getattr(args, "use_ead_a", False))
            or bool(getattr(args, "use_sa_dd", False))
        )
        evidence_path = getattr(args, "mllm_evidence_path", "")
        evidence_enabled = (
            (self.use_mllm_verification and mllm_action not in risk_only_actions)
            or (evidence_needed and bool(evidence_path))
        )
        self.mllm_verifier = MLLMEvidenceVerifier(
            path=evidence_path,
            enabled=evidence_enabled,
            mode=getattr(args, "mllm_verify_mode", "mm"),
            label_map_text=getattr(args, "mllm_label_map", "positive:0,negative:1,neutral:2"),
            missing_policy=getattr(args, "mllm_missing_policy", "reject"),
            shuffle=getattr(args, "mllm_evidence_shuffle", False),
            seed=getattr(args, "seed", 1),
        )
        self.print_fn(self.mllm_verifier.health_summary())
        self.sa_dd_history = {}
        self.sa_dd_class_ema = torch.full(
            (self.num_classes,),
            1.0 / float(self.num_classes),
            dtype=torch.float,
        )
        self.sa_dd_epoch_class_counts = torch.zeros(
            self.num_classes,
            dtype=torch.float,
        )
        self.dctr_external_seen = set()
        self.dctr_internal_seen_epoch = set()
        self.best_epoch = None

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _sa_dd_history_stability(self, sample_ids, current_prob, momentum):
        ids = []
        if torch.is_tensor(sample_ids):
            ids = [str(value) for value in sample_ids.detach().cpu().view(-1).tolist()]
        else:
            for value in sample_ids:
                if torch.is_tensor(value):
                    value = value.item()
                ids.append(str(value))

        current = _normalize_prob(current_prob.detach())
        previous = torch.zeros_like(current)
        has_history = torch.zeros(
            current.shape[0],
            dtype=torch.bool,
            device=current.device,
        )
        for index, sample_id in enumerate(ids):
            old = self.sa_dd_history.get(sample_id)
            if old is not None:
                previous[index] = old.to(current.device)
                has_history[index] = True

        stability = torch.zeros(
            current.shape[0],
            dtype=torch.float,
            device=current.device,
        )
        if has_history.any():
            stability[has_history] = _js_similarity(
                current[has_history],
                previous[has_history],
            )

        momentum = float(momentum)
        current_cpu = current.cpu()
        for index, sample_id in enumerate(ids):
            old = self.sa_dd_history.get(sample_id)
            if old is None:
                updated = current_cpu[index]
            else:
                updated = momentum * old + (1.0 - momentum) * current_cpu[index]
                updated = updated / updated.sum().clamp_min(1e-6)
            self.sa_dd_history[sample_id] = updated.detach()
        return stability.detach(), float(has_history.float().mean().detach().cpu())

    def _sa_dd_update_class_state(
        self,
        pseudo_label,
        momentum,
        max_class_share,
        min_class_share,
        guard_min_count,
    ):
        counts = torch.bincount(
            pseudo_label.detach().cpu(),
            minlength=self.num_classes,
        ).float()
        batch_dist = counts / counts.sum().clamp_min(1.0)
        momentum = float(momentum)
        self.sa_dd_class_ema = (
            momentum * self.sa_dd_class_ema
            + (1.0 - momentum) * batch_dist
        )
        self.sa_dd_class_ema = (
            self.sa_dd_class_ema / self.sa_dd_class_ema.sum().clamp_min(1e-6)
        )
        self.sa_dd_epoch_class_counts += counts
        epoch_total = float(self.sa_dd_epoch_class_counts.sum())
        epoch_dist = (
            self.sa_dd_epoch_class_counts
            / self.sa_dd_epoch_class_counts.sum().clamp_min(1.0)
        )

        collapse = False
        if epoch_total >= float(guard_min_count):
            collapse = bool(
                epoch_dist.max() > float(max_class_share)
                or epoch_dist.min() < float(min_class_share)
                or self.sa_dd_class_ema.max() > float(max_class_share)
                or self.sa_dd_class_ema.min() < float(min_class_share)
            )
        return collapse, epoch_dist.tolist(), self.sa_dd_class_ema.tolist()

    def train(self, args, epoch, best_eval_acc, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        ucrf_unlabeled_scale = 0.0
        ucrf_effective_beta = 0.0
        if getattr(args, "use_ucrf", False):
            ucrf_warmup_epoch = int(
                getattr(args, "ucrf_warmup_epoch", 20)
            )
            ucrf_rampup_epoch = int(
                getattr(args, "ucrf_rampup_epoch", 20)
            )
            if epoch >= ucrf_warmup_epoch:
                if ucrf_rampup_epoch > 0:
                    ucrf_unlabeled_scale = min(
                        1.0,
                        float(epoch - ucrf_warmup_epoch + 1)
                        / float(ucrf_rampup_epoch),
                    )
                else:
                    ucrf_unlabeled_scale = 1.0
            ucrf_effective_beta = (
                float(getattr(args, "ucrf_residual_beta", 0.3))
                * ucrf_unlabeled_scale
            )
            ucrf_model = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
            ucrf_model.set_ucrf_residual_beta(ucrf_effective_beta)
        if getattr(args, "use_sa_dd", False):
            self.sa_dd_epoch_class_counts.zero_()
        if getattr(args, "mllm_action", "veto") == "dctr_plf":
            self.dctr_internal_seen_epoch = set()
        # TODO
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        sup_losses = AverageMeter()
        unsup_losses = AverageMeter()
        decouple_losses = AverageMeter()
        # contrast_losses = AverageMeter()
        total_losses = AverageMeter()
        mask_ratios = AverageMeter()
        # distribution_losses = AverageMeter()
        lr_last = 0
        batch_data_time = AverageMeter()
        batch_model_time = AverageMeter()

        pseudo_true_ratios = AverageMeter()
        mllm_base_coverages = AverageMeter()
        mllm_final_coverages = AverageMeter()
        mllm_weighted_coverages = AverageMeter()
        mllm_retentions = AverageMeter()
        mllm_base_pseudo_accs = AverageMeter()
        mllm_final_pseudo_accs = AverageMeter()
        mllm_weighted_pseudo_accs = AverageMeter()
        mllm_effective_signals = AverageMeter()
        mllm_reject_error_rates = AverageMeter()
        mllm_missing_ratios = AverageMeter()
        mllm_mm_agree_ratios = AverageMeter()
        mllm_internal_conflict_ratios = AverageMeter()
        mllm_external_conflict_ratios = AverageMeter()
        mllm_high_risk_ratios = AverageMeter()
        mllm_used_ratios = AverageMeter()
        mllm_agree_on_risk_ratios = AverageMeter()
        mllm_reject_on_risk_ratios = AverageMeter()
        mllm_low_risk_pseudo_accs = AverageMeter()
        mllm_high_risk_base_pseudo_accs = AverageMeter()
        mllm_high_risk_final_pseudo_accs = AverageMeter()
        mllm_ecs_rescued_coverages = AverageMeter()
        mllm_ecs_rescued_pseudo_accs = AverageMeter()
        mllm_ecs_opposite_conflict_pseudo_accs = AverageMeter()
        mllm_ecs_emotional_consensus_counts = AverageMeter()
        mllm_ecs_neutral_consensus_counts = AverageMeter()
        mllm_ecs_modality_imbalance_counts = AverageMeter()
        mllm_ecs_opposite_conflict_counts = AverageMeter()
        mllm_ecs_qwen_agree_pass_counts = AverageMeter()
        mllm_ecs_qwen_disagree_pass_counts = AverageMeter()
        mllm_hybrid_support_means = AverageMeter()
        mllm_hybrid_q_conf_means = AverageMeter()
        mllm_hybrid_q_margin_means = AverageMeter()
        mllm_hybrid_strong_support_ratios = AverageMeter()
        mllm_hybrid_uncertain_ratios = AverageMeter()
        mllm_hybrid_high_conflict_ratios = AverageMeter()
        mllm_hybrid_opposite_conflict_ratios = AverageMeter()
        mllm_hybrid_neutral_limited_ratios = AverageMeter()
        dctr_reliability_m = AverageMeter()
        dctr_reliability_t = AverageMeter()
        dctr_reliability_v = AverageMeter()
        dctr_external_precision_m = AverageMeter()
        dctr_external_precision_t = AverageMeter()
        dctr_external_precision_v = AverageMeter()
        dctr_internal_precision_m = AverageMeter()
        dctr_internal_precision_t = AverageMeter()
        dctr_internal_precision_v = AverageMeter()
        dctr_aug_stability_m = AverageMeter()
        dctr_aug_stability_t = AverageMeter()
        dctr_aug_stability_v = AverageMeter()
        dctr_distribution_similarity_m = AverageMeter()
        dctr_distribution_similarity_t = AverageMeter()
        dctr_distribution_similarity_v = AverageMeter()
        dctr_weighted_coverage_m = AverageMeter()
        dctr_weighted_coverage_t = AverageMeter()
        dctr_weighted_coverage_v = AverageMeter()
        dctr_weighted_pseudo_acc_m = AverageMeter()
        dctr_weighted_pseudo_acc_t = AverageMeter()
        dctr_weighted_pseudo_acc_v = AverageMeter()
        ce_umc_text_losses = AverageMeter()
        ce_umc_image_losses = AverageMeter()
        ce_umc_text_selected_ratios = AverageMeter()
        ce_umc_image_selected_ratios = AverageMeter()
        ce_umc_text_num_selecteds = AverageMeter()
        ce_umc_image_num_selecteds = AverageMeter()
        ce_umc_text_confidence_means = AverageMeter()
        ce_umc_image_confidence_means = AverageMeter()
        ce_umc_text_certainty_means = AverageMeter()
        ce_umc_image_certainty_means = AverageMeter()
        ce_umc_text_weight_means = AverageMeter()
        ce_umc_image_weight_means = AverageMeter()
        ce_umc_text_agreements = AverageMeter()
        ce_umc_image_agreements = AverageMeter()
        ce_umc_scope_ratios = AverageMeter()
        ce_umc_text_effective_lambdas = AverageMeter()
        ce_umc_image_effective_lambdas = AverageMeter()
        ec_pfd_losses = AverageMeter()
        ec_pfd_mask_ratios = AverageMeter()
        ec_pfd_num_selecteds = AverageMeter()
        ec_pfd_reliability_means = AverageMeter()
        ec_pfd_external_target_ratios = AverageMeter()
        ec_pfd_neutral_uncertain_ratios = AverageMeter()
        ead_a_losses = AverageMeter()
        ead_a_text_anchor_losses = AverageMeter()
        ead_a_image_anchor_losses = AverageMeter()
        ead_a_common_align_losses = AverageMeter()
        ead_a_selected_ratios = AverageMeter()
        ead_a_num_selecteds = AverageMeter()
        ead_a_reliability_means = AverageMeter()
        ead_a_text_conf_means = AverageMeter()
        ead_a_image_conf_means = AverageMeter()
        ead_a_agreement_means = AverageMeter()
        ead_a_opposite_ratios = AverageMeter()
        ead_a_neutral_ratios = AverageMeter()
        ead_a_text_weight_means = AverageMeter()
        ead_a_image_weight_means = AverageMeter()
        ead_a_common_weight_means = AverageMeter()
        ead_a_effective_lambdas = AverageMeter()
        sa_dd_losses = AverageMeter()
        sa_dd_common_losses = AverageMeter()
        sa_dd_private_losses = AverageMeter()
        sa_dd_selected_ratios = AverageMeter()
        sa_dd_num_selecteds = AverageMeter()
        sa_dd_gate_means = AverageMeter()
        sa_dd_history_stability_means = AverageMeter()
        sa_dd_aug_stability_means = AverageMeter()
        sa_dd_relation_agreement_means = AverageMeter()
        sa_dd_relation_conf_means = AverageMeter()
        sa_dd_stability_gate_means = AverageMeter()
        sa_dd_model_uncertainty_means = AverageMeter()
        sa_dd_ecplf_risk_means = AverageMeter()
        sa_dd_information_means = AverageMeter()
        sa_dd_relation_strength_means = AverageMeter()
        sa_dd_same_score_means = AverageMeter()
        sa_dd_opposite_score_means = AverageMeter()
        sa_dd_uncertain_score_means = AverageMeter()
        sa_dd_common_ratio_means = AverageMeter()
        sa_dd_private_ratio_means = AverageMeter()
        sa_dd_selected_accept_ratios = AverageMeter()
        sa_dd_selected_reject_ratios = AverageMeter()
        sa_dd_opposite_ratios = AverageMeter()
        sa_dd_neutral_ratios = AverageMeter()
        sa_dd_collapse_backoffs = AverageMeter()
        sa_dd_history_coverages = AverageMeter()
        sa_dd_effective_lambdas = AverageMeter()
        sa_dd_class_weight_mass = np.zeros(self.num_classes, dtype=np.float64)
        sa_dd_epoch_pred_dist = [0.0 for _ in range(self.num_classes)]
        sa_dd_ema_pred_dist = self.sa_dd_class_ema.tolist()
        msd_losses = AverageMeter()
        msd_labeled_losses = AverageMeter()
        msd_unlabeled_losses = AverageMeter()
        msd_selected_ratios = AverageMeter()
        msd_num_selecteds = AverageMeter()
        msd_reliability_means = AverageMeter()
        msd_valid_evidence_ratios = AverageMeter()
        msd_opposite_conflict_ratios = AverageMeter()
        msd_neutral_consensus_ratios = AverageMeter()
        msd_text_branch_accs = AverageMeter()
        msd_image_branch_accs = AverageMeter()
        msd_multimodal_branch_accs = AverageMeter()
        msd_oracle_branch_accs = AverageMeter()
        msd_selector_branch_accs = AverageMeter()
        msd_prior_selected_branch_accs = AverageMeter()
        msd_effective_lambdas = AverageMeter()
        msd_selector_grad_norms = AverageMeter()
        msd_selector_sum = np.zeros(3, dtype=np.float64)
        msd_prior_sum = np.zeros(3, dtype=np.float64)
        msd_stat_batches = 0
        dctr_msg_losses = AverageMeter()
        dctr_msg_labeled_losses = AverageMeter()
        dctr_msg_unlabeled_losses = AverageMeter()
        dctr_msg_selected_ratios = AverageMeter()
        dctr_msg_num_selecteds = AverageMeter()
        dctr_msg_reliability_means = AverageMeter()
        dctr_msg_reliability_margin_means = AverageMeter()
        dctr_msg_selector_entropies = AverageMeter()
        dctr_msg_prior_entropies = AverageMeter()
        dctr_msg_selector_prior_agreements = AverageMeter()
        dctr_msg_text_branch_accs = AverageMeter()
        dctr_msg_image_branch_accs = AverageMeter()
        dctr_msg_multimodal_branch_accs = AverageMeter()
        dctr_msg_oracle_branch_accs = AverageMeter()
        dctr_msg_selector_branch_accs = AverageMeter()
        dctr_msg_prior_branch_accs = AverageMeter()
        dctr_msg_effective_lambdas = AverageMeter()
        dctr_msg_selector_grad_norms = AverageMeter()
        dctr_msg_selector_sum = np.zeros(3, dtype=np.float64)
        dctr_msg_prior_sum = np.zeros(3, dtype=np.float64)
        dctr_msg_stat_batches = 0
        ucrf_losses = AverageMeter()
        ucrf_labeled_losses = AverageMeter()
        ucrf_unlabeled_losses = AverageMeter()
        ucrf_selected_ratios = AverageMeter()
        ucrf_num_selecteds = AverageMeter()
        ucrf_reliability_means = AverageMeter()
        ucrf_selector_entropies = AverageMeter()
        ucrf_selector_utility_agreements = AverageMeter()
        ucrf_unlabeled_utility_agreements = AverageMeter()
        ucrf_text_candidate_accs = AverageMeter()
        ucrf_image_candidate_accs = AverageMeter()
        ucrf_common_candidate_accs = AverageMeter()
        ucrf_oracle_candidate_accs = AverageMeter()
        ucrf_selector_candidate_accs = AverageMeter()
        ucrf_selector_confidences = AverageMeter()
        ucrf_selector_grad_norms = AverageMeter()
        ucrf_selector_sum = np.zeros(3, dtype=np.float64)
        ucrf_stat_batches = 0
        mllm_final_pred_counts = np.zeros(self.num_classes, dtype=np.int64)
        mllm_weighted_final_pred_sums = np.zeros(self.num_classes, dtype=np.float64)
        diag_confidences = []
        diag_selected_confidences = []
        diag_unselected_confidences = []
        diag_total_count = 0
        diag_candidate_count = 0
        diag_masked_count = 0
        diag_masked_correct = 0.0
        diag_weight_sum = 0.0
        diag_weighted_correct = 0.0
        diag_all_correct = 0.0

        start_batch.record()

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args, epoch=epoch)
            print(eval_dict)
        # focal_loss = MultiClassFocalLossWithAlpha()
        iter_num = 0
        for (x_lb_idx, x_lb, t_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s0, x_ulb_s1, t_ulb, y_ulb) in tqdm(zip(self.loader_dict['train_lb'],
                                                                     self.loader_dict['train_ulb']), total=len(self.loader_dict['train_ulb'])):
            # break
            iter_num += 1
            end_batch.record()
            torch.cuda.synchronize()
            batch_data_time.update(start_batch.elapsed_time(end_batch) / 1000)
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s0.shape[0] and num_ulb == x_ulb_s1.shape[0]
            try:
                x_lb.cuda(args.gpu)
            except Exception as e:
                print("An error occurred:", e)           
            x_lb, x_ulb_w, x_ulb_s0, x_ulb_s1 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s0.cuda(args.gpu), x_ulb_s1.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            img_inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s0, x_ulb_s1))
            def _to_text_list(items):
                out = []
                for item in items:
                    out.append(item.item() if hasattr(item, "item") else item)
                return out

            t_lb = _to_text_list(t_lb)
            t_ulb = _to_text_list(t_ulb)
            # TODO text的输入格式
            text_input = t_lb + t_ulb + t_ulb + t_ulb
            text_input = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True)
            text_input = {key: value.cuda(args.gpu) for key, value in text_input.items()}

            # hyper-params for update
            # T = self.t_fn(self.it)
            # p_cutoff = self.p_fn(self.it)

            # inference and calculate sup/unsup losses
            with amp_cm():
                # logits, features = self.model(img_inputs, text_input)
                # res = self.fusion_model(img_inputs, text_input)
                output = self.model(img_inputs, text_input)
                # logits_m = output['pre_m']
                logits_m = output['pre_m_att']
                logits_common = output.get('logits_common', output['pre_m'])
                logits_t = output['pre_t']
                logits_v = output['pre_v']

                features_m = output['c_l'] + output['c_v']
                features_t = output['c_l']
                features_v = output['c_v']

                logits_x_lb = logits_m[:num_lb]
                logits_t_lb = logits_t[:num_lb] 
                logits_v_lb = logits_v[:num_lb] 
                # logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                logits_x_ulb_w, logits_x_ulb_s0, logits_x_ulb_s1 = torch.split(logits_m[num_lb:], num_ulb)
                logits_common_ulb_w, logits_common_ulb_s0, logits_common_ulb_s1 = torch.split(logits_common[num_lb:], num_ulb)
                logits_t_ulb_w, logits_t_ulb_s0, logits_t_ulb_s1 = torch.split(logits_t[num_lb:], num_ulb)
                logits_v_ulb_w, logits_v_ulb_s0, logits_v_ulb_s1 = torch.split(logits_v[num_lb:], num_ulb)
                # Keep gradient-bearing weak-view logits for CE-UMC before the
                # pseudo-label block detaches its local prediction tensors.
                logits_t_ulb_w_ce_umc = logits_t_ulb_w
                logits_v_ulb_w_ce_umc = logits_v_ulb_w
                modal_support = output["modal_support"]
                modal_support_lb = modal_support[:num_lb]
                modal_support_ulb_w, modal_support_ulb_s0, modal_support_ulb_s1 = torch.split(
                    modal_support[num_lb:],
                    num_ulb,
                )
                candidate_logits = output["candidate_logits"]
                candidate_logits_lb = candidate_logits[:num_lb]
                (
                    candidate_logits_ulb_w,
                    candidate_logits_ulb_s0,
                    candidate_logits_ulb_s1,
                ) = torch.split(candidate_logits[num_lb:], num_ulb)
                ucrf_selector_confidence = output[
                    "ucrf_selector_confidence"
                ]
                ucrf_selector_confidence_lb = ucrf_selector_confidence[:num_lb]

                features_lb = features_m[:num_lb]
                features_ulb_w, features_ulb_s0, features_ulb_s1 = torch.split(features_m[num_lb:], num_ulb)
                # features_t_ulb_w, features_c_ulb_s0, features_c_ulb_s1 = torch.split(features_t[num_lb:], num_ulb)
                # features_v_ulb_w, features_v_ulb_s0, features_v_ulb_s1 = torch.split(features_v[num_lb:], num_ulb)
                
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                pre_v_in_m = output['pre_v_in_m'][:num_lb]
                pre_t_in_m = output['pre_t_in_m'][:num_lb]
                
                if epoch <= 10:
                    select_v = []
                    select_t = []
                    v_sup_loss = 0
                    t_sup_loss = 0
                    label_filter_threshold = 0.5 * sup_loss
                    for i in range(len(pre_v_in_m)):
                        if(ce_loss(pre_v_in_m[i], y_lb[i]) < label_filter_threshold):
                            select_v.append(i)
                        if(ce_loss(pre_t_in_m[i], y_lb[i]) < label_filter_threshold):
                            select_t.append(i)
                    if select_v != []:
                        v_sup_loss =  ce_loss(logits_v_lb[select_v], y_lb[select_v], reduction='mean')  
                    if select_t != []:    
                        t_sup_loss =  ce_loss(logits_t_lb[select_t], y_lb[select_t], reduction='mean')   
                    sup_loss = sup_loss + v_sup_loss + t_sup_loss             


                modal_index = output['modal_index']
                modal_index_x_ulb_w, _, _ = torch.split(modal_index[num_lb:], num_ulb)
                pre_m_in_t = output['pre_m_in_t']
                pre_m_in_t,_,_ = torch.split(pre_m_in_t[num_lb:], num_ulb)
                pre_m_in_v = output['pre_m_in_v']
                pre_m_in_v,_,_ = torch.split(pre_m_in_v[num_lb:], num_ulb)

                with torch.no_grad():
                    logits_x_ulb_w = logits_x_ulb_w.detach()
                    logits_t_ulb_w = logits_t_ulb_w.detach()
                    logits_v_ulb_w = logits_v_ulb_w.detach()

                    modal_index_x_ulb_w = modal_index_x_ulb_w.detach()

                    features_lb = features_lb.detach()
                    features_ulb_w = features_ulb_w.detach()  # [bs*,2816]
                    # features_t_ulb_w = features_t_ulb_w.detach()
                    # features_v_ulb_w = features_v_ulb_w.detach()

                    pre_m_in_t = pre_m_in_t.detach()
                    pre_m_in_v = pre_m_in_v.detach()

                    ulb_probs = torch.softmax(logits_x_ulb_w, dim=1)
                    t_ulb_probs = torch.softmax(logits_t_ulb_w, dim=1)
                    v_ulb_probs = torch.softmax(logits_v_ulb_w, dim=1)
                    
                    scores, lbs_u_guess = torch.max(ulb_probs, dim=1)
                    t_scores, t_lbs_u_guess = torch.max(t_ulb_probs, dim=1)
                    v_scores, v_lbs_u_guess = torch.max(v_ulb_probs, dim=1)
                    confidence_scores = scores.clone()

                    if epoch <= 10:
                        persudo_filter_threshold = 1 * sup_loss
                        for i,idx in enumerate(modal_index_x_ulb_w):
                            if idx == 2:
                                loss_m_in_t = ce_loss(pre_m_in_t[i], t_lbs_u_guess[i])
                                loss_m_in_v = ce_loss(pre_m_in_v[i], v_lbs_u_guess[i])
                                if loss_m_in_t > persudo_filter_threshold or loss_m_in_v > persudo_filter_threshold:
                                    scores[i] = 0

                    threshold = args.threshold
                    confidence_mask = confidence_scores.ge(threshold)
                    mask = scores.ge(threshold)
                    t_mask = t_scores.ge(threshold)
                    v_mask = v_scores.ge(threshold)
                    mllm_sample_weights = torch.ones_like(scores, dtype=torch.float)
                    dctr_sample_weights_t = torch.ones_like(
                        scores,
                        dtype=torch.float,
                    )
                    dctr_sample_weights_v = torch.ones_like(
                        scores,
                        dtype=torch.float,
                    )
                    dctr_reliability = None
                    unlabeled_evidence = None
                    dctr_log = None
                    loss_ec_pfd = scores.new_tensor(0.0)
                    ec_pfd_log = {
                        "loss_ec_pfd": 0.0,
                        "ec_pfd_mask_ratio": 0.0,
                        "ec_pfd_num_selected": 0,
                        "ec_pfd_reliability_mean": 0.0,
                        "ec_pfd_external_target_ratio": 0.0,
                        "ec_pfd_neutral_uncertain_ratio": 0.0,
                    }
                    loss_ead_a = scores.new_tensor(0.0)
                    ead_a_log = {
                        "loss_ead_a": 0.0,
                        "ead_a_text_anchor_loss": 0.0,
                        "ead_a_image_anchor_loss": 0.0,
                        "ead_a_common_align_loss": 0.0,
                        "ead_a_selected_ratio": 0.0,
                        "ead_a_num_selected": 0,
                        "ead_a_reliability_mean": 0.0,
                        "ead_a_text_conf_mean": 0.0,
                        "ead_a_image_conf_mean": 0.0,
                        "ead_a_agreement_mean": 0.0,
                        "ead_a_opposite_ratio": 0.0,
                        "ead_a_neutral_ratio": 0.0,
                        "ead_a_text_weight_mean": 0.0,
                        "ead_a_image_weight_mean": 0.0,
                        "ead_a_common_weight_mean": 0.0,
                        "ead_a_effective_lambda": 0.0,
                    }
                    lambda_ead_a_eff = 0.0
                    loss_sa_dd = scores.new_tensor(0.0)
                    sa_dd_log = {
                        "loss_sa_dd": 0.0,
                        "sa_dd_common_loss": 0.0,
                        "sa_dd_private_loss": 0.0,
                        "sa_dd_selected_ratio": 0.0,
                        "sa_dd_num_selected": 0,
                        "sa_dd_gate_mean": 0.0,
                        "sa_dd_history_stability_mean": 0.0,
                        "sa_dd_aug_stability_mean": 0.0,
                        "sa_dd_relation_agreement_mean": 0.0,
                        "sa_dd_relation_conf_mean": 0.0,
                        "sa_dd_stability_gate_mean": 0.0,
                        "sa_dd_model_uncertainty_mean": 0.0,
                        "sa_dd_ecplf_risk_mean": 0.0,
                        "sa_dd_information_mean": 0.0,
                        "sa_dd_relation_strength_mean": 0.0,
                        "sa_dd_same_score_mean": 0.0,
                        "sa_dd_opposite_score_mean": 0.0,
                        "sa_dd_uncertain_score_mean": 0.0,
                        "sa_dd_common_ratio_mean": 0.0,
                        "sa_dd_private_ratio_mean": 0.0,
                        "sa_dd_selected_accept_ratio": 0.0,
                        "sa_dd_selected_reject_ratio": 0.0,
                        "sa_dd_opposite_ratio": 0.0,
                        "sa_dd_neutral_ratio": 0.0,
                        "sa_dd_collapse_backoff": 0.0,
                        "sa_dd_class_weight_mass": [
                            0.0 for _ in range(self.num_classes)
                        ],
                        "sa_dd_history_coverage": 0.0,
                        "sa_dd_effective_lambda": 0.0,
                    }
                    lambda_sa_dd_eff = 0.0
                    loss_msd = scores.new_tensor(0.0)
                    msd_log = {
                        "loss_msd": 0.0,
                        "loss_msd_labeled": 0.0,
                        "loss_msd_unlabeled": 0.0,
                        "msd_selected_ratio": 0.0,
                        "msd_num_selected": 0,
                        "msd_reliability_mean": 0.0,
                        "msd_valid_evidence_ratio": 0.0,
                        "msd_opposite_conflict_ratio": 0.0,
                        "msd_neutral_consensus_ratio": 0.0,
                        "msd_text_branch_acc": 0.0,
                        "msd_image_branch_acc": 0.0,
                        "msd_multimodal_branch_acc": 0.0,
                        "msd_oracle_branch_acc": 0.0,
                        "msd_selector_branch_acc": 0.0,
                        "msd_prior_selected_branch_acc": 0.0,
                        "msd_selector_mean": [0.0, 0.0, 0.0],
                        "msd_prior_mean": [0.0, 0.0, 0.0],
                        "msd_effective_lambda": 0.0,
                    }
                    lambda_msd_eff = 0.0
                    loss_dctr_msg = scores.new_tensor(0.0)
                    dctr_msg_log = {
                        "loss_dctr_msg": 0.0,
                        "loss_dctr_msg_labeled": 0.0,
                        "loss_dctr_msg_unlabeled": 0.0,
                        "dctr_msg_selected_ratio": 0.0,
                        "dctr_msg_num_selected": 0,
                        "dctr_msg_reliability_mean": 0.0,
                        "dctr_msg_reliability_margin_mean": 0.0,
                        "dctr_msg_selector_entropy": 0.0,
                        "dctr_msg_prior_entropy": 0.0,
                        "dctr_msg_selector_prior_agreement": 0.0,
                        "dctr_msg_text_branch_acc": 0.0,
                        "dctr_msg_image_branch_acc": 0.0,
                        "dctr_msg_multimodal_branch_acc": 0.0,
                        "dctr_msg_oracle_branch_acc": 0.0,
                        "dctr_msg_selector_branch_acc": 0.0,
                        "dctr_msg_prior_branch_acc": 0.0,
                        "dctr_msg_selector_mean": [0.0, 0.0, 0.0],
                        "dctr_msg_prior_mean": [0.0, 0.0, 0.0],
                        "dctr_msg_effective_lambda": 0.0,
                    }
                    lambda_dctr_msg_eff = 0.0
                    y_ulb = y_ulb.cuda(args.gpu)
                    risk_features_needed = self.use_mllm_verification
                    if risk_features_needed:
                        branch_conflict = (
                            t_lbs_u_guess.ne(v_lbs_u_guess)
                            | t_lbs_u_guess.ne(lbs_u_guess)
                            | v_lbs_u_guess.ne(lbs_u_guess)
                        )
                        top2_scores = torch.topk(ulb_probs, k=2, dim=1).values
                        margin = top2_scores[:, 0] - top2_scores[:, 1]
                        low_margin = margin.lt(
                            getattr(args, "risk_margin_threshold", 0.2)
                        )

                        strong_probs = torch.softmax(
                            logits_x_ulb_s0.detach(), dim=1
                        )
                        weak_strong_kl = F.kl_div(
                            torch.log(strong_probs.clamp_min(1e-12)),
                            ulb_probs,
                            reduction="none",
                        ).sum(dim=1)
                        unstable_aug = weak_strong_kl.gt(
                            getattr(args, "risk_kl_threshold", 0.5)
                        )

                        pos_id = self.mllm_verifier.label_map["positive"]
                        neg_id = self.mllm_verifier.label_map["negative"]
                        modality_conflict = (
                            (t_lbs_u_guess.eq(pos_id) & v_lbs_u_guess.eq(neg_id))
                            | (
                                t_lbs_u_guess.eq(neg_id)
                                & v_lbs_u_guess.eq(pos_id)
                            )
                        )
                        high_risk = (
                            branch_conflict
                            | low_margin
                            | unstable_aug
                            | modality_conflict
                        )

                    if self.use_mllm_verification:
                        mask_before_mllm = mask.clone()
                        t_mask_before_mllm = t_mask.clone()
                        v_mask_before_mllm = v_mask.clone()
                        base_lbs_u_guess = lbs_u_guess.clone()

                        if getattr(args, "mllm_selective_verify", True):
                            if getattr(args, "mllm_verify_policy", "risky_only") == "all":
                                verify_mask = torch.ones_like(mask_before_mllm, dtype=torch.bool)
                            else:
                                verify_mask = high_risk
                        else:
                            verify_mask = torch.ones_like(mask_before_mllm, dtype=torch.bool)

                        mask, mllm_log, mllm_sample_weights, override_labels = self.mllm_verifier.verify(
                            base_mask=mask_before_mllm,
                            sample_ids=x_ulb_idx,
                            y_t=t_lbs_u_guess,
                            y_v=v_lbs_u_guess,
                            y_m=lbs_u_guess,
                            y_true=y_ulb,
                            verify_mask=verify_mask,
                            action=getattr(args, "mllm_action", "veto"),
                            soft_weight=getattr(args, "mllm_soft_weight", 0.5),
                            ecs_conf_threshold=getattr(args, "mllm_ecs_conf_threshold", 0.75),
                            hybrid_config={
                                "support_high": getattr(args, "hybrid_support_high", 0.65),
                                "support_low": getattr(args, "hybrid_support_low", 0.30),
                                "qwen_conf_high": getattr(args, "hybrid_qwen_conf_high", 0.75),
                                "weight_agree": getattr(args, "hybrid_weight_agree", 1.0),
                                "weight_uncertain": getattr(args, "hybrid_weight_uncertain", 0.5),
                                "weight_high_conflict": getattr(args, "hybrid_weight_high_conflict", 0.0),
                                "weight_opposite_conflict": getattr(args, "hybrid_weight_opposite_conflict", 0.0),
                                "neutral_max_weight": getattr(args, "hybrid_neutral_max_weight", 0.5),
                                "use_distribution": getattr(args, "hybrid_use_distribution", True),
                                "use_opposite_conflict": getattr(args, "hybrid_use_opposite_conflict", True),
                            },
                        )
                        if getattr(args, "mllm_action", "veto") == "override":
                            lbs_u_guess = override_labels
                        if getattr(args, "mllm_action", "veto") == "dctr_plf":
                            dctr_model = (
                                self.model.module
                                if hasattr(self.model, "module")
                                else self.model
                            )
                            calibrator = getattr(
                                dctr_model,
                                "dctr_calibrator",
                                None,
                            )
                            if calibrator is None:
                                raise RuntimeError(
                                    "DCTR-PLF action requires DMD.dctr_calibrator"
                                )

                            labeled_evidence = self.mllm_verifier.dctr_evidence(
                                sample_ids=x_lb_idx,
                                device=y_lb.device,
                            )
                            labeled_probs = torch.stack(
                                [
                                    torch.softmax(logits_x_lb.detach(), dim=1),
                                    torch.softmax(logits_t_lb.detach(), dim=1),
                                    torch.softmax(logits_v_lb.detach(), dim=1),
                                ],
                                dim=1,
                            )
                            if torch.is_tensor(x_lb_idx):
                                labeled_ids = [
                                    str(value)
                                    for value in x_lb_idx.detach()
                                    .cpu()
                                    .view(-1)
                                    .tolist()
                                ]
                            else:
                                labeled_ids = []
                                for value in x_lb_idx:
                                    if torch.is_tensor(value):
                                        value = value.item()
                                    labeled_ids.append(str(value))
                            external_update_mask = torch.tensor(
                                [
                                    sample_id not in self.dctr_external_seen
                                    for sample_id in labeled_ids
                                ],
                                dtype=torch.bool,
                                device=y_lb.device,
                            )
                            internal_update_mask = torch.tensor(
                                [
                                    sample_id
                                    not in self.dctr_internal_seen_epoch
                                    for sample_id in labeled_ids
                                ],
                                dtype=torch.bool,
                                device=y_lb.device,
                            )
                            calibrator.update(
                                model_labels=labeled_probs.argmax(dim=-1),
                                evidence_labels=labeled_evidence["labels"],
                                relation_group=labeled_evidence["relation_group"],
                                y_true=y_lb.detach(),
                                evidence_valid=labeled_evidence["valid"],
                                external_update_mask=external_update_mask,
                                internal_update_mask=internal_update_mask,
                            )
                            self.dctr_external_seen.update(labeled_ids)
                            self.dctr_internal_seen_epoch.update(labeled_ids)

                            unlabeled_evidence = self.mllm_verifier.dctr_evidence(
                                sample_ids=x_ulb_idx,
                                device=lbs_u_guess.device,
                            )
                            weak_branch_probs = torch.stack(
                                [ulb_probs, t_ulb_probs, v_ulb_probs],
                                dim=1,
                            )
                            strong_branch_probs = torch.stack(
                                [
                                    torch.softmax(
                                        logits_x_ulb_s0.detach(),
                                        dim=1,
                                    ),
                                    torch.softmax(
                                        logits_t_ulb_s0.detach(),
                                        dim=1,
                                    ),
                                    torch.softmax(
                                        logits_v_ulb_s0.detach(),
                                        dim=1,
                                    ),
                                ],
                                dim=1,
                            )
                            dctr_reliability, dctr_parts = (
                                compute_dctr_reliability(
                                    calibrator=calibrator,
                                    weak_probs=weak_branch_probs,
                                    strong_probs=strong_branch_probs,
                                    evidence=unlabeled_evidence,
                                    use_aug_stability=getattr(
                                        args,
                                        "dctr_use_aug_stability",
                                        True,
                                    ),
                                    use_js_similarity=getattr(
                                        args,
                                        "dctr_use_js_similarity",
                                        True,
                                    ),
                                )
                            )
                            if getattr(args, "dctr_apply_all", True):
                                dctr_apply_mask = torch.ones_like(
                                    verify_mask,
                                    dtype=torch.bool,
                                )
                            else:
                                dctr_apply_mask = verify_mask
                            dctr_reliability = torch.where(
                                dctr_apply_mask.view(-1, 1),
                                dctr_reliability,
                                torch.ones_like(dctr_reliability),
                            )

                            mllm_sample_weights = dctr_reliability[:, 0]
                            dctr_sample_weights_t = dctr_reliability[:, 1]
                            dctr_sample_weights_v = dctr_reliability[:, 2]
                            min_reliability = float(
                                getattr(args, "dctr_min_reliability", 0.05)
                            )
                            mask = (
                                mask_before_mllm
                                & mllm_sample_weights.ge(min_reliability)
                            )
                            t_mask = (
                                t_mask_before_mllm
                                & dctr_sample_weights_t.ge(min_reliability)
                            )
                            v_mask = (
                                v_mask_before_mllm
                                & dctr_sample_weights_v.ge(min_reliability)
                            )

                            calibrated_log = (
                                self.mllm_verifier.diagnostic_log(
                                    base_mask=mask_before_mllm,
                                    final_mask=mask,
                                    labels=lbs_u_guess,
                                    y_true=y_ulb,
                                    loss_weight=mllm_sample_weights,
                                )
                            )
                            mllm_log.update(calibrated_log)
                            mllm_log["mllm_used_ratio"] = float(
                                dctr_apply_mask.float().mean().detach().cpu()
                            )

                            dctr_branch_masks = torch.stack(
                                [mask, t_mask, v_mask],
                                dim=1,
                            )
                            dctr_branch_weights = dctr_reliability * (
                                dctr_branch_masks.float()
                            )
                            dctr_branch_labels = torch.stack(
                                [
                                    lbs_u_guess,
                                    t_lbs_u_guess,
                                    v_lbs_u_guess,
                                ],
                                dim=1,
                            )
                            dctr_correct = dctr_branch_labels.eq(
                                y_ulb.view(-1, 1)
                            ).float()
                            dctr_weight_sums = dctr_branch_weights.sum(dim=0)
                            dctr_weighted_acc = (
                                dctr_branch_weights * dctr_correct
                            ).sum(dim=0) / dctr_weight_sums.clamp_min(1e-7)
                            dctr_log = {
                                "reliability": dctr_reliability.mean(dim=0),
                                "external_precision": dctr_parts[
                                    "external_precision"
                                ].mean(dim=0),
                                "internal_precision": dctr_parts[
                                    "internal_precision"
                                ].mean(dim=0),
                                "aug_stability": dctr_parts[
                                    "aug_stability"
                                ].mean(dim=0),
                                "distribution_similarity": dctr_parts[
                                    "distribution_similarity"
                                ].mean(dim=0),
                                "weighted_coverage": (
                                    dctr_weight_sums
                                    / float(max(num_ulb, 1))
                                ),
                                "weighted_pseudo_acc": dctr_weighted_acc,
                            }
                        else:
                            # Legacy modes share the multimodal decision with
                            # both private branches. Keep their behavior intact.
                            t_mask = t_mask & mask
                            v_mask = v_mask & mask

                        correct_base = base_lbs_u_guess.eq(y_ulb)
                        correct_final = lbs_u_guess.eq(y_ulb)

                        def update_masked_meter(meter, values, select):
                            count = int(select.sum().detach().cpu())
                            if count > 0:
                                meter.update(float(values[select].float().mean().detach().cpu()), count)

                        update_masked_meter(
                            mllm_low_risk_pseudo_accs,
                            correct_base,
                            mask_before_mllm & (~high_risk),
                        )
                        update_masked_meter(
                            mllm_high_risk_base_pseudo_accs,
                            correct_base,
                            mask_before_mllm & high_risk,
                        )
                        update_masked_meter(
                            mllm_high_risk_final_pseudo_accs,
                            correct_final,
                            mask & high_risk,
                        )
                        mllm_base_coverages.update(mllm_log["base_coverage"])
                        mllm_final_coverages.update(mllm_log["final_coverage"])
                        mllm_weighted_coverages.update(mllm_log["weighted_coverage"])
                        mllm_retentions.update(mllm_log["retention"])
                        mllm_base_pseudo_accs.update(mllm_log["base_pseudo_acc"])
                        mllm_final_pseudo_accs.update(mllm_log["final_pseudo_acc"])
                        mllm_weighted_pseudo_accs.update(mllm_log["weighted_pseudo_acc"])
                        mllm_effective_signals.update(mllm_log["effective_signal"])
                        mllm_reject_error_rates.update(mllm_log["reject_error_rate"])
                        mllm_missing_ratios.update(mllm_log["missing_ratio"])
                        mllm_mm_agree_ratios.update(mllm_log["mm_agree_ratio"])
                        mllm_internal_conflict_ratios.update(mllm_log["internal_conflict_ratio"])
                        mllm_external_conflict_ratios.update(mllm_log["external_conflict_ratio"])
                        mllm_high_risk_ratios.update(float(high_risk.float().mean().detach().cpu()))
                        mllm_used_ratios.update(mllm_log["mllm_used_ratio"])
                        mllm_agree_on_risk_ratios.update(mllm_log["mllm_agree_ratio_on_verified"])
                        mllm_reject_on_risk_ratios.update(mllm_log["mllm_reject_ratio_on_verified"])
                        mllm_ecs_rescued_coverages.update(mllm_log["ecs_rescued_coverage"])
                        mllm_ecs_rescued_pseudo_accs.update(mllm_log["ecs_rescued_pseudo_acc"])
                        mllm_ecs_opposite_conflict_pseudo_accs.update(mllm_log["ecs_opposite_conflict_pseudo_acc"])
                        mllm_ecs_emotional_consensus_counts.update(mllm_log["ecs_emotional_consensus_count"])
                        mllm_ecs_neutral_consensus_counts.update(mllm_log["ecs_neutral_consensus_count"])
                        mllm_ecs_modality_imbalance_counts.update(mllm_log["ecs_modality_imbalance_count"])
                        mllm_ecs_opposite_conflict_counts.update(mllm_log["ecs_opposite_conflict_count"])
                        mllm_ecs_qwen_agree_pass_counts.update(mllm_log["ecs_qwen_agree_pass_count"])
                        mllm_ecs_qwen_disagree_pass_counts.update(mllm_log["ecs_qwen_disagree_pass_count"])
                        mllm_hybrid_support_means.update(mllm_log["hybrid_support_mean"])
                        mllm_hybrid_q_conf_means.update(mllm_log["hybrid_q_conf_mean"])
                        mllm_hybrid_q_margin_means.update(mllm_log["hybrid_q_margin_mean"])
                        mllm_hybrid_strong_support_ratios.update(mllm_log["hybrid_strong_support_ratio"])
                        mllm_hybrid_uncertain_ratios.update(mllm_log["hybrid_uncertain_ratio"])
                        mllm_hybrid_high_conflict_ratios.update(mllm_log["hybrid_high_conflict_ratio"])
                        mllm_hybrid_opposite_conflict_ratios.update(mllm_log["hybrid_opposite_conflict_ratio"])
                        mllm_hybrid_neutral_limited_ratios.update(mllm_log["hybrid_neutral_limited_ratio"])
                        if dctr_log is not None:
                            branch_values = [
                                (
                                    dctr_reliability_m,
                                    dctr_external_precision_m,
                                    dctr_internal_precision_m,
                                    dctr_aug_stability_m,
                                    dctr_distribution_similarity_m,
                                    dctr_weighted_coverage_m,
                                    dctr_weighted_pseudo_acc_m,
                                ),
                                (
                                    dctr_reliability_t,
                                    dctr_external_precision_t,
                                    dctr_internal_precision_t,
                                    dctr_aug_stability_t,
                                    dctr_distribution_similarity_t,
                                    dctr_weighted_coverage_t,
                                    dctr_weighted_pseudo_acc_t,
                                ),
                                (
                                    dctr_reliability_v,
                                    dctr_external_precision_v,
                                    dctr_internal_precision_v,
                                    dctr_aug_stability_v,
                                    dctr_distribution_similarity_v,
                                    dctr_weighted_coverage_v,
                                    dctr_weighted_pseudo_acc_v,
                                ),
                            ]
                            for branch, meters in enumerate(branch_values):
                                values = (
                                    dctr_log["reliability"][branch],
                                    dctr_log["external_precision"][branch],
                                    dctr_log["internal_precision"][branch],
                                    dctr_log["aug_stability"][branch],
                                    dctr_log["distribution_similarity"][branch],
                                    dctr_log["weighted_coverage"][branch],
                                    dctr_log["weighted_pseudo_acc"][branch],
                                )
                                for meter, value in zip(meters, values):
                                    meter.update(float(value.detach().cpu()))
                        if mask.any():
                            mllm_final_pred_counts += np.bincount(
                                lbs_u_guess[mask].detach().cpu().numpy(),
                                minlength=self.num_classes,
                            )
                        weight_np = (mask.float() * mllm_sample_weights.float()).detach().cpu().numpy()
                        label_np = lbs_u_guess.detach().cpu().numpy()
                        for cls in range(self.num_classes):
                            mllm_weighted_final_pred_sums[cls] += float(weight_np[label_np == cls].sum())

                    pseudo_true_ratios.update(((lbs_u_guess == y_ulb) * mask).sum() / (mask.sum()+1e-7))

                if getattr(args, "log_pseudo_diag", True):
                    pseudo_correct_diag = lbs_u_guess.eq(y_ulb).detach()
                    final_weight_diag = (
                        mask.float() * mllm_sample_weights.detach().float()
                    )
                    selected_diag = final_weight_diag.gt(0)
                    confidence_diag = confidence_scores.detach().float()

                    diag_total_count += int(num_ulb)
                    diag_candidate_count += int(
                        confidence_mask.sum().detach().cpu()
                    )
                    diag_all_correct += float(
                        pseudo_correct_diag.float().sum().detach().cpu()
                    )
                    diag_masked_count += int(selected_diag.sum().detach().cpu())
                    diag_masked_correct += float(
                        pseudo_correct_diag[selected_diag]
                        .float()
                        .sum()
                        .detach()
                        .cpu()
                    )
                    diag_weight_sum += float(
                        final_weight_diag.sum().detach().cpu()
                    )
                    diag_weighted_correct += float(
                        (
                            final_weight_diag
                            * pseudo_correct_diag.float()
                        ).sum().detach().cpu()
                    )
                    diag_confidences.extend(
                        confidence_diag.cpu().numpy().tolist()
                    )
                    if selected_diag.any():
                        diag_selected_confidences.extend(
                            confidence_diag[selected_diag].cpu().numpy().tolist()
                        )
                    if (~selected_diag).any():
                        diag_unselected_confidences.extend(
                            confidence_diag[~selected_diag].cpu().numpy().tolist()
                        )

                loss_ce_umc_text = logits_t_ulb_w_ce_umc.new_tensor(0.0)
                loss_ce_umc_image = logits_v_ulb_w_ce_umc.new_tensor(0.0)
                lambda_ce_umc_text_eff = 0.0
                lambda_ce_umc_image_eff = 0.0
                ce_umc_log = {
                    "loss_ce_umc_text": 0.0,
                    "loss_ce_umc_image": 0.0,
                    "ce_umc_scope_ratio": 0.0,
                    "ce_umc_text_valid_ratio": 0.0,
                    "ce_umc_image_valid_ratio": 0.0,
                    "ce_umc_text_selected_ratio": 0.0,
                    "ce_umc_image_selected_ratio": 0.0,
                    "ce_umc_text_num_selected": 0,
                    "ce_umc_image_num_selected": 0,
                    "ce_umc_text_confidence_mean": 0.0,
                    "ce_umc_image_confidence_mean": 0.0,
                    "ce_umc_text_certainty_mean": 0.0,
                    "ce_umc_image_certainty_mean": 0.0,
                    "ce_umc_text_weight_mean": 0.0,
                    "ce_umc_image_weight_mean": 0.0,
                    "ce_umc_text_target_entropy_mean": 0.0,
                    "ce_umc_image_target_entropy_mean": 0.0,
                    "ce_umc_text_model_evidence_agreement": 0.0,
                    "ce_umc_image_model_evidence_agreement": 0.0,
                }
                if getattr(args, "use_ce_umc", False):
                    ce_umc_warmup = int(
                        getattr(args, "ce_umc_warmup_epoch", 10)
                    )
                    ce_umc_rampup = int(
                        getattr(args, "ce_umc_rampup_epoch", 20)
                    )
                    ce_umc_scale = 0.0
                    if epoch >= ce_umc_warmup:
                        if ce_umc_rampup > 0:
                            ce_umc_scale = min(
                                1.0,
                                float(epoch - ce_umc_warmup + 1)
                                / float(ce_umc_rampup),
                            )
                        else:
                            ce_umc_scale = 1.0
                    lambda_ce_umc_text_eff = (
                        float(getattr(args, "lambda_ce_umc_text", 0.02))
                        * ce_umc_scale
                    )
                    lambda_ce_umc_image_eff = (
                        float(getattr(args, "lambda_ce_umc_image", 0.05))
                        * ce_umc_scale
                    )

                    if (
                        lambda_ce_umc_text_eff > 0.0
                        or lambda_ce_umc_image_eff > 0.0
                    ):
                        qwen_ce_umc = self.mllm_verifier.msd_evidence(
                            sample_ids=x_ulb_idx,
                            device=lbs_u_guess.device,
                        )
                        if getattr(
                            args,
                            "ce_umc_evidence_shuffle",
                            False,
                        ):
                            ce_umc_permutation = torch.randperm(
                                num_ulb,
                                device=lbs_u_guess.device,
                            )
                            qwen_ce_umc = {
                                key: value[ce_umc_permutation]
                                for key, value in qwen_ce_umc.items()
                            }

                        ce_umc_scope = getattr(
                            args,
                            "ce_umc_scope",
                            "all",
                        )
                        if ce_umc_scope == "all":
                            ce_umc_scope_mask = torch.ones_like(
                                mask,
                                dtype=torch.bool,
                            )
                        elif ce_umc_scope == "plf_accepted":
                            ce_umc_scope_mask = mask.detach().bool()
                        elif ce_umc_scope == "plf_rejected":
                            ce_umc_scope_mask = ~mask.detach().bool()
                        else:
                            raise ValueError(
                                f"Unknown CE-UMC scope: {ce_umc_scope}"
                            )

                        (
                            loss_ce_umc_text,
                            loss_ce_umc_image,
                            ce_umc_stats,
                        ) = compute_ce_umc_loss(
                            logits_text=logits_t_ulb_w_ce_umc,
                            logits_image=logits_v_ulb_w_ce_umc,
                            qwen_evidence=qwen_ce_umc,
                            scope_mask=ce_umc_scope_mask,
                            target_temperature=getattr(
                                args,
                                "ce_umc_target_temperature",
                                2.0,
                            ),
                            text_conf_threshold=getattr(
                                args,
                                "ce_umc_text_conf_threshold",
                                0.90,
                            ),
                            image_conf_threshold=getattr(
                                args,
                                "ce_umc_image_conf_threshold",
                                0.90,
                            ),
                            min_certainty=getattr(
                                args,
                                "ce_umc_min_certainty",
                                0.20,
                            ),
                            confidence_power=getattr(
                                args,
                                "ce_umc_confidence_power",
                                1.0,
                            ),
                            min_samples=getattr(
                                args,
                                "ce_umc_min_samples",
                                2,
                            ),
                        )
                        ce_umc_log.update(ce_umc_stats)

                if getattr(args, "use_ead_a", False):
                    ead_a_base_lambda = float(getattr(args, "lambda_ead_a", 0.02))
                    ead_a_warmup_epoch = int(getattr(args, "ead_a_warmup_epoch", 30))
                    if ead_a_warmup_epoch > 0:
                        ead_a_ramp = min(
                            1.0,
                            float(epoch + 1) / float(ead_a_warmup_epoch),
                        )
                    else:
                        ead_a_ramp = 1.0
                    lambda_ead_a_eff = ead_a_base_lambda * ead_a_ramp
                    ead_a_log["ead_a_effective_lambda"] = lambda_ead_a_eff

                    if lambda_ead_a_eff > 0.0:
                        qwen_ead_a = self.mllm_verifier.msd_evidence(
                            sample_ids=x_ulb_idx,
                            device=lbs_u_guess.device,
                        )
                        if getattr(args, "ead_a_evidence_shuffle", False):
                            ead_a_permutation = torch.randperm(
                                num_ulb,
                                device=lbs_u_guess.device,
                            )
                            qwen_ead_a = {
                                key: value[ead_a_permutation]
                                for key, value in qwen_ead_a.items()
                            }

                        logits_text_recon_s0 = torch.split(
                            output["logits_text_recon"][num_lb:],
                            num_ulb,
                        )[1]
                        logits_image_recon_s0 = torch.split(
                            output["logits_image_recon"][num_lb:],
                            num_ulb,
                        )[1]
                        common_text_s0 = torch.split(
                            output["c_l"][num_lb:],
                            num_ulb,
                        )[1]
                        common_image_s0 = torch.split(
                            output["c_v"][num_lb:],
                            num_ulb,
                        )[1]
                        ead_a_reliability = (
                            scores.detach().float()
                            * mllm_sample_weights.detach().float()
                        ).clamp(min=0.0, max=1.0)
                        loss_ead_a, ead_a_stats, _ = compute_ead_a_loss(
                            logits_text_recon=logits_text_recon_s0,
                            logits_image_recon=logits_image_recon_s0,
                            common_text=common_text_s0,
                            common_image=common_image_s0,
                            qwen_evidence=qwen_ead_a,
                            reliability=ead_a_reliability,
                            accept_mask=mask,
                            reliability_threshold=getattr(
                                args,
                                "ead_a_reliability_threshold",
                                0.45,
                            ),
                            min_samples=getattr(args, "ead_a_min_samples", 2),
                            common_align_weight=getattr(
                                args,
                                "ead_a_common_align_weight",
                                0.2,
                            ),
                            conflict_boost=getattr(
                                args,
                                "ead_a_conflict_boost",
                                0.5,
                            ),
                            neutral_scale=getattr(
                                args,
                                "ead_a_neutral_scale",
                                0.5,
                            ),
                            opposite_common_scale=getattr(
                                args,
                                "ead_a_opposite_common_scale",
                                0.0,
                            ),
                        )
                        ead_a_log.update(ead_a_stats)
                        ead_a_log["ead_a_effective_lambda"] = lambda_ead_a_eff

                if getattr(args, "use_sa_dd", False):
                    history_stability, history_coverage = self._sa_dd_history_stability(
                        sample_ids=x_ulb_idx,
                        current_prob=ulb_probs,
                        momentum=getattr(args, "sa_dd_history_momentum", 0.9),
                    )
                    collapse_backoff, sa_dd_epoch_pred_dist, sa_dd_ema_pred_dist = (
                        self._sa_dd_update_class_state(
                            pseudo_label=lbs_u_guess,
                            momentum=getattr(args, "sa_dd_class_momentum", 0.95),
                            max_class_share=getattr(args, "sa_dd_max_class_share", 0.85),
                            min_class_share=getattr(args, "sa_dd_min_class_share", 0.02),
                            guard_min_count=getattr(
                                args,
                                "sa_dd_class_guard_min_count",
                                64,
                            ),
                        )
                    )
                    sa_dd_log["sa_dd_history_coverage"] = history_coverage
                    sa_dd_log["sa_dd_collapse_backoff"] = float(collapse_backoff)

                    sa_dd_warmup = int(getattr(args, "sa_dd_warmup_epoch", 30))
                    sa_dd_rampup = int(getattr(args, "sa_dd_rampup_epoch", 20))
                    if epoch >= sa_dd_warmup:
                        if sa_dd_rampup > 0:
                            sa_dd_ramp = min(
                                1.0,
                                float(epoch - sa_dd_warmup + 1)
                                / float(sa_dd_rampup),
                            )
                        else:
                            sa_dd_ramp = 1.0
                        lambda_sa_dd_eff = (
                            float(getattr(args, "lambda_sa_dd", 0.01))
                            * sa_dd_ramp
                        )
                    sa_dd_log["sa_dd_effective_lambda"] = lambda_sa_dd_eff

                    if lambda_sa_dd_eff > 0.0:
                        qwen_sa_dd = self.mllm_verifier.msd_evidence(
                            sample_ids=x_ulb_idx,
                            device=lbs_u_guess.device,
                        )
                        if getattr(args, "sa_dd_evidence_shuffle", False):
                            sa_dd_permutation = torch.randperm(
                                num_ulb,
                                device=lbs_u_guess.device,
                            )
                            qwen_sa_dd = {
                                key: value[sa_dd_permutation]
                                for key, value in qwen_sa_dd.items()
                            }

                        common_text_s0 = torch.split(
                            output["c_l"][num_lb:],
                            num_ulb,
                        )[1]
                        common_image_s0 = torch.split(
                            output["c_v"][num_lb:],
                            num_ulb,
                        )[1]
                        private_text_s0 = torch.split(
                            output["s_l"][num_lb:],
                            num_ulb,
                        )[1]
                        private_image_s0 = torch.split(
                            output["s_v"][num_lb:],
                            num_ulb,
                        )[1]
                        sa_dd_strong_prob = torch.softmax(
                            logits_x_ulb_s0.detach(),
                            dim=1,
                        )
                        loss_sa_dd, sa_dd_stats, _ = compute_sa_dd_loss(
                            common_text=common_text_s0,
                            common_image=common_image_s0,
                            private_text=private_text_s0,
                            private_image=private_image_s0,
                            weak_prob=ulb_probs,
                            strong_prob=sa_dd_strong_prob,
                            qwen_evidence=qwen_sa_dd,
                            history_stability=history_stability,
                            pseudo_label=lbs_u_guess.detach(),
                            ecplf_accept=mask.detach(),
                            ecplf_weight=mllm_sample_weights.detach(),
                            num_classes=self.num_classes,
                            min_history_stability=getattr(
                                args,
                                "sa_dd_min_history_stability",
                                0.6,
                            ),
                            min_aug_stability=getattr(
                                args,
                                "sa_dd_min_aug_stability",
                                0.6,
                            ),
                            min_relation_conf=getattr(
                                args,
                                "sa_dd_min_relation_conf",
                                0.2,
                            ),
                            relation_floor=getattr(
                                args,
                                "sa_dd_relation_floor",
                                0.2,
                            ),
                            relation_ceiling=getattr(
                                args,
                                "sa_dd_relation_ceiling",
                                0.8,
                            ),
                            common_weight=getattr(
                                args,
                                "sa_dd_common_weight",
                                1.0,
                            ),
                            private_weight=getattr(
                                args,
                                "sa_dd_private_weight",
                                1.0,
                            ),
                            ecplf_floor=getattr(
                                args,
                                "sa_dd_ecplf_floor",
                                0.5,
                            ),
                            neutral_scale=getattr(
                                args,
                                "sa_dd_neutral_scale",
                                0.5,
                            ),
                            min_samples=getattr(args, "sa_dd_min_samples", 2),
                            collapse_backoff=collapse_backoff,
                            version=getattr(args, "sa_dd_version", "v2"),
                            positive_id=self.mllm_verifier.positive_id,
                            negative_id=self.mllm_verifier.negative_id,
                            v2_min_gate=getattr(
                                args,
                                "sa_dd_v2_min_gate",
                                0.05,
                            ),
                            v2_stability_temperature=getattr(
                                args,
                                "sa_dd_v2_stability_temperature",
                                0.1,
                            ),
                            ablation=getattr(args, "sa_dd_ablation", "none"),
                            fixed_gate_value=getattr(
                                args,
                                "sa_dd_fixed_gate_value",
                                0.5,
                            ),
                        )
                        sa_dd_log.update(sa_dd_stats)
                        sa_dd_log["sa_dd_history_coverage"] = history_coverage
                        sa_dd_log["sa_dd_effective_lambda"] = lambda_sa_dd_eff

                if getattr(args, "use_msd", False):
                    msd_base_lambda = float(getattr(args, "lambda_msd", 0.1))
                    msd_warmup_epoch = int(getattr(args, "msd_warmup_epoch", 20))
                    msd_rampup_epoch = int(getattr(args, "msd_rampup_epoch", 20))
                    if epoch >= msd_warmup_epoch:
                        if msd_rampup_epoch > 0:
                            msd_ramp = min(
                                1.0,
                                float(epoch - msd_warmup_epoch + 1) / float(msd_rampup_epoch),
                            )
                        else:
                            msd_ramp = 1.0
                        lambda_msd_eff = msd_base_lambda * msd_ramp
                    msd_log["msd_effective_lambda"] = lambda_msd_eff

                    if lambda_msd_eff > 0.0:
                        qwen_msd = self.mllm_verifier.msd_evidence(
                            sample_ids=x_ulb_idx,
                            device=lbs_u_guess.device,
                        )
                        if getattr(args, "msd_evidence_shuffle", False):
                            msd_permutation = torch.randperm(num_ulb, device=lbs_u_guess.device)
                            qwen_msd = {
                                key: value[msd_permutation]
                                for key, value in qwen_msd.items()
                            }
                        msd_reliability = (
                            scores.detach().float() * mllm_sample_weights.detach().float()
                        ).clamp(min=0.0, max=1.0)
                        common_ulb_probs = torch.softmax(logits_common_ulb_w.detach(), dim=1)
                        loss_msd, msd_stats, _ = compute_msd_loss(
                            selector_lb=modal_support_lb,
                            logits_text_lb=logits_t_lb,
                            logits_image_lb=logits_v_lb,
                            logits_multimodal_lb=logits_common[:num_lb],
                            y_lb=y_lb,
                            selector_ulb=modal_support_ulb_w,
                            text_prob_ulb=t_ulb_probs,
                            image_prob_ulb=v_ulb_probs,
                            multimodal_prob_ulb=common_ulb_probs,
                            qwen_evidence=qwen_msd,
                            reliability=msd_reliability,
                            accept_mask=mask,
                            reliability_threshold=getattr(args, "msd_reliability_threshold", 0.75),
                            support_temperature=getattr(args, "msd_support_temperature", 0.5),
                            labeled_temperature=getattr(args, "msd_labeled_temperature", 0.5),
                            smoothing=getattr(args, "msd_support_smoothing", 0.05),
                            labeled_weight=getattr(args, "msd_labeled_weight", 1.0),
                            unlabeled_weight=getattr(args, "msd_unlabeled_weight", 1.0),
                            use_labeled_anchor=getattr(args, "msd_use_labeled_anchor", True),
                            use_unlabeled_evidence=getattr(args, "msd_use_unlabeled_evidence", True),
                            exclude_opposite_conflict=getattr(args, "msd_exclude_opposite_conflict", True),
                            exclude_neutral_consensus=getattr(args, "msd_exclude_neutral_consensus", False),
                            prior_mode=getattr(args, "msd_prior_mode", "evidence"),
                            y_ulb=y_ulb.detach(),
                        )
                        msd_log.update(msd_stats)
                        msd_log["msd_effective_lambda"] = lambda_msd_eff

                if getattr(args, "use_dctr_msg", False):
                    dctr_msg_base_lambda = float(
                        getattr(args, "lambda_dctr_msg", 0.03)
                    )
                    dctr_msg_warmup_epoch = int(
                        getattr(args, "dctr_msg_warmup_epoch", 20)
                    )
                    dctr_msg_rampup_epoch = int(
                        getattr(args, "dctr_msg_rampup_epoch", 20)
                    )
                    if epoch >= dctr_msg_warmup_epoch:
                        if dctr_msg_rampup_epoch > 0:
                            dctr_msg_ramp = min(
                                1.0,
                                float(epoch - dctr_msg_warmup_epoch + 1)
                                / float(dctr_msg_rampup_epoch),
                            )
                        else:
                            dctr_msg_ramp = 1.0
                        lambda_dctr_msg_eff = (
                            dctr_msg_base_lambda * dctr_msg_ramp
                        )
                    dctr_msg_log["dctr_msg_effective_lambda"] = (
                        lambda_dctr_msg_eff
                    )

                    if lambda_dctr_msg_eff > 0.0:
                        if dctr_reliability is None:
                            raise RuntimeError(
                                "DCTR-MSG requires DCTR reliability from "
                                "--mllm_action dctr_plf"
                            )
                        common_ulb_probs = torch.softmax(
                            logits_common_ulb_w.detach(),
                            dim=1,
                        )
                        opposite_conflict = None
                        if unlabeled_evidence is not None:
                            opposite_conflict = unlabeled_evidence[
                                "opposite_conflict"
                            ]
                        dctr_msg_reliability = dctr_reliability
                        if getattr(
                            args,
                            "dctr_msg_reliability_shuffle",
                            False,
                        ):
                            dctr_msg_permutation = torch.randperm(
                                num_ulb,
                                device=dctr_reliability.device,
                            )
                            dctr_msg_reliability = dctr_reliability[
                                dctr_msg_permutation
                            ]
                            if opposite_conflict is not None:
                                opposite_conflict = opposite_conflict[
                                    dctr_msg_permutation
                                ]
                        (
                            loss_dctr_msg,
                            dctr_msg_stats,
                            _,
                        ) = compute_dctr_msg_loss(
                            selector_lb=modal_support_lb,
                            logits_text_lb=logits_t_lb,
                            logits_image_lb=logits_v_lb,
                            logits_multimodal_lb=logits_common[:num_lb],
                            y_lb=y_lb,
                            selector_ulb=modal_support_ulb_w,
                            text_prob_ulb=t_ulb_probs,
                            image_prob_ulb=v_ulb_probs,
                            multimodal_prob_ulb=common_ulb_probs,
                            dctr_reliability=dctr_msg_reliability,
                            accept_mask=mask,
                            opposite_conflict=opposite_conflict,
                            reliability_threshold=getattr(
                                args,
                                "dctr_msg_reliability_threshold",
                                0.45,
                            ),
                            prior_temperature=getattr(
                                args,
                                "dctr_msg_prior_temperature",
                                0.5,
                            ),
                            labeled_temperature=getattr(
                                args,
                                "dctr_msg_labeled_temperature",
                                0.5,
                            ),
                            smoothing=getattr(
                                args,
                                "dctr_msg_smoothing",
                                0.05,
                            ),
                            min_margin=getattr(
                                args,
                                "dctr_msg_min_margin",
                                0.02,
                            ),
                            min_samples=getattr(
                                args,
                                "dctr_msg_min_samples",
                                2,
                            ),
                            labeled_weight=getattr(
                                args,
                                "dctr_msg_labeled_weight",
                                1.0,
                            ),
                            unlabeled_weight=getattr(
                                args,
                                "dctr_msg_unlabeled_weight",
                                1.0,
                            ),
                            use_labeled_anchor=getattr(
                                args,
                                "dctr_msg_use_labeled_anchor",
                                True,
                            ),
                            exclude_opposite_conflict=getattr(
                                args,
                                "dctr_msg_exclude_opposite_conflict",
                                False,
                            ),
                            y_ulb=y_ulb.detach(),
                        )
                        dctr_msg_log.update(dctr_msg_stats)
                        dctr_msg_log["dctr_msg_effective_lambda"] = (
                            lambda_dctr_msg_eff
                        )

                loss_ucrf = scores.new_tensor(0.0)
                ucrf_log = {
                    "loss_ucrf": 0.0,
                    "loss_ucrf_labeled": 0.0,
                    "loss_ucrf_unlabeled": 0.0,
                    "ucrf_selected_ratio": 0.0,
                    "ucrf_num_selected": 0,
                    "ucrf_reliability_mean": 0.0,
                    "ucrf_selector_entropy": 0.0,
                    "ucrf_selector_utility_agreement": 0.0,
                    "ucrf_unlabeled_utility_agreement": 0.0,
                    "ucrf_text_candidate_acc": 0.0,
                    "ucrf_image_candidate_acc": 0.0,
                    "ucrf_common_candidate_acc": 0.0,
                    "ucrf_oracle_candidate_acc": 0.0,
                    "ucrf_selector_candidate_acc": 0.0,
                    "ucrf_selector_mean": [0.0, 0.0, 0.0],
                }
                if getattr(args, "use_ucrf", False):
                    if dctr_reliability is not None:
                        ucrf_reliability = dctr_reliability[:, 0]
                    else:
                        ucrf_reliability = (
                            scores.detach().float()
                            * mllm_sample_weights.detach().float()
                        ).clamp(0.0, 1.0)
                    ucrf_unlabeled_weight = 0.0
                    if getattr(args, "ucrf_use_unlabeled", True):
                        ucrf_unlabeled_weight = (
                            float(
                                getattr(
                                    args,
                                    "ucrf_unlabeled_weight",
                                    0.5,
                                )
                            )
                            * ucrf_unlabeled_scale
                        )
                    (
                        loss_ucrf,
                        ucrf_stats,
                        _,
                    ) = compute_ucrf_loss(
                        selector_lb=modal_support_lb,
                        candidate_logits_lb=candidate_logits_lb,
                        y_lb=y_lb,
                        selector_ulb=modal_support_ulb_w,
                        candidate_logits_ulb=candidate_logits_ulb_w,
                        pseudo_label=lbs_u_guess,
                        reliability=ucrf_reliability,
                        accept_mask=mask,
                        utility_temperature=getattr(
                            args,
                            "ucrf_utility_temperature",
                            0.5,
                        ),
                        reliability_threshold=getattr(
                            args,
                            "ucrf_reliability_threshold",
                            0.75,
                        ),
                        min_samples=getattr(
                            args,
                            "ucrf_min_samples",
                            2,
                        ),
                        labeled_weight=getattr(
                            args,
                            "ucrf_labeled_weight",
                            1.0,
                        ),
                        unlabeled_weight=ucrf_unlabeled_weight,
                    )
                    ucrf_log.update(ucrf_stats)

                if getattr(args, "use_ec_pfd", False):
                    reliability_score = (scores.detach().float() * mllm_sample_weights.detach().float()).clamp(min=0.0, max=1.0)
                    ec_pfd_pseudo_prob, ec_pfd_neutral_uncertain, ec_pfd_target_log = self.mllm_verifier.ec_pfd_targets(
                        sample_ids=x_ulb_idx,
                        y_m=lbs_u_guess.detach(),
                        fallback_prob=ulb_probs.detach(),
                        device=lbs_u_guess.device,
                    )
                    loss_ec_pfd, ec_pfd_stats, _ = compute_ec_pfd_loss(
                        logits_common=logits_common_ulb_s0,
                        logits_text_private=logits_t_ulb_s0,
                        logits_image_private=logits_v_ulb_s0,
                        pseudo_label=lbs_u_guess,
                        reliability_score=reliability_score,
                        accept_mask=mask,
                        threshold=getattr(args, "ec_pfd_threshold", 0.75),
                        pseudo_prob=ec_pfd_pseudo_prob,
                        neutral_uncertain=ec_pfd_neutral_uncertain,
                        use_soft_label=getattr(args, "ec_pfd_use_soft_label", True),
                        detach_target=getattr(args, "ec_pfd_detach_target", True),
                        min_samples=getattr(args, "ec_pfd_min_samples", 1),
                    )
                    ec_pfd_log.update(ec_pfd_stats)
                    ec_pfd_log.update(ec_pfd_target_log)

                if (
                    self.use_mllm_verification
                    and getattr(args, "mllm_action", "veto")
                    in ("hybrid_cerw", "dctr_plf")
                ):
                    ce_m = F.cross_entropy(logits_x_ulb_s0, lbs_u_guess, reduction='none')
                    ce_t = F.cross_entropy(logits_t_ulb_s0, t_lbs_u_guess, reduction='none')
                    ce_v = F.cross_entropy(logits_v_ulb_s0, v_lbs_u_guess, reduction='none')
                    weight_m = mask.float() * mllm_sample_weights.float()
                    if getattr(args, "mllm_action", "veto") == "dctr_plf":
                        weight_t = (
                            t_mask.float()
                            * dctr_sample_weights_t.float()
                        )
                        weight_v = (
                            v_mask.float()
                            * dctr_sample_weights_v.float()
                        )
                    else:
                        weight_t = t_mask.float() * mllm_sample_weights.float()
                        weight_v = v_mask.float() * mllm_sample_weights.float()
                    unsup_loss_m = (ce_m * weight_m).sum() / (weight_m.sum() + 1e-7)
                    unsup_loss_t = (ce_t * weight_t).sum() / (weight_t.sum() + 1e-7)
                    unsup_loss_v = (ce_v * weight_v).sum() / (weight_v.sum() + 1e-7)
                    unsup_loss = unsup_loss_m + unsup_loss_t + unsup_loss_v
                else:
                    unsup_loss_m = F.cross_entropy(logits_x_ulb_s0, lbs_u_guess, reduction='none') * mask * mllm_sample_weights
                    unsup_loss_t = F.cross_entropy(logits_t_ulb_s0, t_lbs_u_guess, reduction='none') * t_mask * mllm_sample_weights
                    unsup_loss_v = F.cross_entropy(logits_v_ulb_s0, v_lbs_u_guess, reduction='none') * v_mask * mllm_sample_weights
                    unsup_loss = unsup_loss_m.mean() + unsup_loss_t.mean() + unsup_loss_v.mean()

                # decouple loss
                # reconstruction loss
                loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                loss_recon = loss_recon_l + loss_recon_v

                # cycle consistency loss between s_x and s_x_r
                loss_sl_slr = self.MSE(output['s_l'], output['s_l_r'])
                loss_sv_slv = self.MSE(output['s_v'], output['s_v_r'])
                loss_s_sr = loss_sl_slr + loss_sv_slv

                # ort loss
                cosine_similarity_s_c_l = self.cosine(output['s_l'], output['c_l'],
                                                        torch.tensor([-1]).cuda()).mean(0)
                cosine_similarity_s_c_v = self.cosine(output['s_v'], output['c_v'],
                                                        torch.tensor([-1]).cuda()).mean(0)
                loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v
                # margin loss
                c_l, c_v = output['c_l_sim'], output['c_v_sim']
                c_l_lb, c_v_lb = c_l[:num_lb], c_v[:num_lb]
                c_l_ulb_w, c_l_ulb_s0, c_l_ulb_s1 = torch.split(c_l[num_lb:], num_ulb)
                c_v_ulb_w, c_v_ulb_s0, c_v_ulb_s1 = torch.split(c_v[num_lb:], num_ulb)
                ids, feats = [], []
                for i in range(y_lb.size(0)):
                    feats.append(c_l_lb[i].view(1, -1))
                    feats.append(c_v_lb[i].view(1, -1))
                    ids.append(y_lb[i].view(1, -1))
                    ids.append(y_lb[i].view(1, -1))

                sa_dd_v1_replacement_active = (
                    getattr(args, "use_sa_dd", False)
                    and str(getattr(args, "sa_dd_version", "v2")).lower()
                    == "v1"
                    and getattr(
                        args,
                        "sa_dd_replace_unlabeled_sim",
                        True,
                    )
                    and lambda_sa_dd_eff > 0.0
                    and sa_dd_log["sa_dd_num_selected"]
                    >= int(getattr(args, "sa_dd_min_samples", 2))
                    and sa_dd_log["sa_dd_collapse_backoff"] < 0.5
                )
                if sa_dd_v1_replacement_active:
                    # SA-DD replaces the pseudo-label-driven unlabeled common
                    # similarity term only in the legacy v1 compatibility path.
                    sim_indices = []
                elif self.use_mllm_verification and getattr(args, "mllm_gate_sim_loss", True):
                    sim_indices = torch.where(mask)[0].detach().cpu().tolist()
                else:
                    sim_indices = range(y_ulb.size(0))

                for i in sim_indices:
                    feats.append(c_l_ulb_w[i].view(1, -1))
                    feats.append(c_v_ulb_w[i].view(1, -1))
                    ids.append(lbs_u_guess[i].view(1, -1))
                    ids.append(lbs_u_guess[i].view(1, -1))               
                feats = torch.cat(feats, dim=0)
                ids = torch.cat(ids, dim=0)
                loss_sim = self.sim_loss(ids, feats)

                decouple_loss = loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1

                total_loss = (
                    sup_loss
                    + self.lambda_u * unsup_loss
                    + decouple_loss
                    + getattr(args, "lambda_ec_pfd", 0.2) * loss_ec_pfd
                    + lambda_ead_a_eff * loss_ead_a
                    + lambda_sa_dd_eff * loss_sa_dd
                    + lambda_msd_eff * loss_msd
                    + lambda_dctr_msg_eff * loss_dctr_msg
                    + getattr(args, "lambda_ucrf", 0.05) * loss_ucrf
                    + lambda_ce_umc_text_eff * loss_ce_umc_text
                    + lambda_ce_umc_image_eff * loss_ce_umc_image
                )
                # only attention
                # total_loss = sup_loss + self.lambda_u * unsup_loss_m.mean() + decouple_loss
                # only SMC
                # total_loss = sup_loss + self.lambda_u * unsup_loss + decouple_loss
                # base
                # total_loss = sup_loss + self.lambda_u * unsup_loss_m.mean()
                

            sup_losses.update(sup_loss.cpu().detach())
            unsup_losses.update(unsup_loss.cpu().detach())
            decouple_losses.update(decouple_loss.cpu().detach())
            if getattr(args, "use_ce_umc", False):
                ce_umc_text_losses.update(
                    ce_umc_log["loss_ce_umc_text"]
                )
                ce_umc_image_losses.update(
                    ce_umc_log["loss_ce_umc_image"]
                )
                ce_umc_text_selected_ratios.update(
                    ce_umc_log["ce_umc_text_selected_ratio"]
                )
                ce_umc_image_selected_ratios.update(
                    ce_umc_log["ce_umc_image_selected_ratio"]
                )
                ce_umc_text_num_selecteds.update(
                    ce_umc_log["ce_umc_text_num_selected"]
                )
                ce_umc_image_num_selecteds.update(
                    ce_umc_log["ce_umc_image_num_selected"]
                )
                ce_umc_text_confidence_means.update(
                    ce_umc_log["ce_umc_text_confidence_mean"]
                )
                ce_umc_image_confidence_means.update(
                    ce_umc_log["ce_umc_image_confidence_mean"]
                )
                ce_umc_text_certainty_means.update(
                    ce_umc_log["ce_umc_text_certainty_mean"]
                )
                ce_umc_image_certainty_means.update(
                    ce_umc_log["ce_umc_image_certainty_mean"]
                )
                ce_umc_text_weight_means.update(
                    ce_umc_log["ce_umc_text_weight_mean"]
                )
                ce_umc_image_weight_means.update(
                    ce_umc_log["ce_umc_image_weight_mean"]
                )
                ce_umc_text_agreements.update(
                    ce_umc_log[
                        "ce_umc_text_model_evidence_agreement"
                    ]
                )
                ce_umc_image_agreements.update(
                    ce_umc_log[
                        "ce_umc_image_model_evidence_agreement"
                    ]
                )
                ce_umc_scope_ratios.update(
                    ce_umc_log["ce_umc_scope_ratio"]
                )
                ce_umc_text_effective_lambdas.update(
                    lambda_ce_umc_text_eff
                )
                ce_umc_image_effective_lambdas.update(
                    lambda_ce_umc_image_eff
                )
            if getattr(args, "use_ec_pfd", False):
                ec_pfd_losses.update(ec_pfd_log["loss_ec_pfd"])
                ec_pfd_mask_ratios.update(ec_pfd_log["ec_pfd_mask_ratio"])
                ec_pfd_num_selecteds.update(ec_pfd_log["ec_pfd_num_selected"])
                ec_pfd_reliability_means.update(ec_pfd_log["ec_pfd_reliability_mean"])
                ec_pfd_external_target_ratios.update(ec_pfd_log["ec_pfd_external_target_ratio"])
                ec_pfd_neutral_uncertain_ratios.update(ec_pfd_log["ec_pfd_neutral_uncertain_ratio"])
            if getattr(args, "use_ead_a", False):
                ead_a_losses.update(ead_a_log["loss_ead_a"])
                ead_a_text_anchor_losses.update(ead_a_log["ead_a_text_anchor_loss"])
                ead_a_image_anchor_losses.update(ead_a_log["ead_a_image_anchor_loss"])
                ead_a_common_align_losses.update(ead_a_log["ead_a_common_align_loss"])
                ead_a_selected_ratios.update(ead_a_log["ead_a_selected_ratio"])
                ead_a_num_selecteds.update(ead_a_log["ead_a_num_selected"])
                ead_a_reliability_means.update(ead_a_log["ead_a_reliability_mean"])
                ead_a_text_conf_means.update(ead_a_log["ead_a_text_conf_mean"])
                ead_a_image_conf_means.update(ead_a_log["ead_a_image_conf_mean"])
                ead_a_agreement_means.update(ead_a_log["ead_a_agreement_mean"])
                ead_a_opposite_ratios.update(ead_a_log["ead_a_opposite_ratio"])
                ead_a_neutral_ratios.update(ead_a_log["ead_a_neutral_ratio"])
                ead_a_text_weight_means.update(ead_a_log["ead_a_text_weight_mean"])
                ead_a_image_weight_means.update(ead_a_log["ead_a_image_weight_mean"])
                ead_a_common_weight_means.update(ead_a_log["ead_a_common_weight_mean"])
                ead_a_effective_lambdas.update(ead_a_log["ead_a_effective_lambda"])
            if getattr(args, "use_sa_dd", False):
                sa_dd_losses.update(sa_dd_log["loss_sa_dd"])
                sa_dd_common_losses.update(sa_dd_log["sa_dd_common_loss"])
                sa_dd_private_losses.update(sa_dd_log["sa_dd_private_loss"])
                sa_dd_selected_ratios.update(sa_dd_log["sa_dd_selected_ratio"])
                sa_dd_num_selecteds.update(sa_dd_log["sa_dd_num_selected"])
                sa_dd_gate_means.update(sa_dd_log["sa_dd_gate_mean"])
                sa_dd_history_stability_means.update(
                    sa_dd_log["sa_dd_history_stability_mean"]
                )
                sa_dd_aug_stability_means.update(
                    sa_dd_log["sa_dd_aug_stability_mean"]
                )
                sa_dd_relation_agreement_means.update(
                    sa_dd_log["sa_dd_relation_agreement_mean"]
                )
                sa_dd_relation_conf_means.update(
                    sa_dd_log["sa_dd_relation_conf_mean"]
                )
                sa_dd_stability_gate_means.update(
                    sa_dd_log["sa_dd_stability_gate_mean"]
                )
                sa_dd_model_uncertainty_means.update(
                    sa_dd_log["sa_dd_model_uncertainty_mean"]
                )
                sa_dd_ecplf_risk_means.update(
                    sa_dd_log["sa_dd_ecplf_risk_mean"]
                )
                sa_dd_information_means.update(
                    sa_dd_log["sa_dd_information_mean"]
                )
                sa_dd_relation_strength_means.update(
                    sa_dd_log["sa_dd_relation_strength_mean"]
                )
                sa_dd_same_score_means.update(
                    sa_dd_log["sa_dd_same_score_mean"]
                )
                sa_dd_opposite_score_means.update(
                    sa_dd_log["sa_dd_opposite_score_mean"]
                )
                sa_dd_uncertain_score_means.update(
                    sa_dd_log["sa_dd_uncertain_score_mean"]
                )
                sa_dd_common_ratio_means.update(
                    sa_dd_log["sa_dd_common_ratio_mean"]
                )
                sa_dd_private_ratio_means.update(
                    sa_dd_log["sa_dd_private_ratio_mean"]
                )
                sa_dd_selected_accept_ratios.update(
                    sa_dd_log["sa_dd_selected_accept_ratio"]
                )
                sa_dd_selected_reject_ratios.update(
                    sa_dd_log["sa_dd_selected_reject_ratio"]
                )
                sa_dd_opposite_ratios.update(sa_dd_log["sa_dd_opposite_ratio"])
                sa_dd_neutral_ratios.update(sa_dd_log["sa_dd_neutral_ratio"])
                sa_dd_collapse_backoffs.update(
                    sa_dd_log["sa_dd_collapse_backoff"]
                )
                sa_dd_history_coverages.update(
                    sa_dd_log["sa_dd_history_coverage"]
                )
                sa_dd_effective_lambdas.update(
                    sa_dd_log["sa_dd_effective_lambda"]
                )
                sa_dd_class_weight_mass += np.asarray(
                    sa_dd_log["sa_dd_class_weight_mass"],
                    dtype=np.float64,
                )
            if getattr(args, "use_msd", False):
                msd_losses.update(msd_log["loss_msd"])
                msd_labeled_losses.update(msd_log["loss_msd_labeled"])
                msd_unlabeled_losses.update(msd_log["loss_msd_unlabeled"])
                msd_selected_ratios.update(msd_log["msd_selected_ratio"])
                msd_num_selecteds.update(msd_log["msd_num_selected"])
                msd_reliability_means.update(msd_log["msd_reliability_mean"])
                msd_valid_evidence_ratios.update(msd_log["msd_valid_evidence_ratio"])
                msd_opposite_conflict_ratios.update(msd_log["msd_opposite_conflict_ratio"])
                msd_neutral_consensus_ratios.update(msd_log["msd_neutral_consensus_ratio"])
                msd_text_branch_accs.update(msd_log["msd_text_branch_acc"])
                msd_image_branch_accs.update(msd_log["msd_image_branch_acc"])
                msd_multimodal_branch_accs.update(msd_log["msd_multimodal_branch_acc"])
                msd_oracle_branch_accs.update(msd_log["msd_oracle_branch_acc"])
                msd_selector_branch_accs.update(msd_log["msd_selector_branch_acc"])
                msd_prior_selected_branch_accs.update(msd_log["msd_prior_selected_branch_acc"])
                msd_effective_lambdas.update(msd_log["msd_effective_lambda"])
                msd_selector_sum += np.asarray(msd_log["msd_selector_mean"], dtype=np.float64)
                msd_prior_sum += np.asarray(msd_log["msd_prior_mean"], dtype=np.float64)
                msd_stat_batches += 1
            if getattr(args, "use_dctr_msg", False):
                dctr_msg_losses.update(dctr_msg_log["loss_dctr_msg"])
                dctr_msg_labeled_losses.update(
                    dctr_msg_log["loss_dctr_msg_labeled"]
                )
                dctr_msg_unlabeled_losses.update(
                    dctr_msg_log["loss_dctr_msg_unlabeled"]
                )
                dctr_msg_selected_ratios.update(
                    dctr_msg_log["dctr_msg_selected_ratio"]
                )
                dctr_msg_num_selecteds.update(
                    dctr_msg_log["dctr_msg_num_selected"]
                )
                dctr_msg_reliability_means.update(
                    dctr_msg_log["dctr_msg_reliability_mean"]
                )
                dctr_msg_reliability_margin_means.update(
                    dctr_msg_log["dctr_msg_reliability_margin_mean"]
                )
                dctr_msg_selector_entropies.update(
                    dctr_msg_log["dctr_msg_selector_entropy"]
                )
                dctr_msg_prior_entropies.update(
                    dctr_msg_log["dctr_msg_prior_entropy"]
                )
                dctr_msg_selector_prior_agreements.update(
                    dctr_msg_log["dctr_msg_selector_prior_agreement"]
                )
                dctr_msg_text_branch_accs.update(
                    dctr_msg_log["dctr_msg_text_branch_acc"]
                )
                dctr_msg_image_branch_accs.update(
                    dctr_msg_log["dctr_msg_image_branch_acc"]
                )
                dctr_msg_multimodal_branch_accs.update(
                    dctr_msg_log["dctr_msg_multimodal_branch_acc"]
                )
                dctr_msg_oracle_branch_accs.update(
                    dctr_msg_log["dctr_msg_oracle_branch_acc"]
                )
                dctr_msg_selector_branch_accs.update(
                    dctr_msg_log["dctr_msg_selector_branch_acc"]
                )
                dctr_msg_prior_branch_accs.update(
                    dctr_msg_log["dctr_msg_prior_branch_acc"]
                )
                dctr_msg_effective_lambdas.update(
                    dctr_msg_log["dctr_msg_effective_lambda"]
                )
                dctr_msg_selector_sum += np.asarray(
                    dctr_msg_log["dctr_msg_selector_mean"],
                    dtype=np.float64,
                )
                dctr_msg_prior_sum += np.asarray(
                    dctr_msg_log["dctr_msg_prior_mean"],
                    dtype=np.float64,
                )
                dctr_msg_stat_batches += 1
            if getattr(args, "use_ucrf", False):
                ucrf_losses.update(ucrf_log["loss_ucrf"])
                ucrf_labeled_losses.update(
                    ucrf_log["loss_ucrf_labeled"]
                )
                ucrf_unlabeled_losses.update(
                    ucrf_log["loss_ucrf_unlabeled"]
                )
                ucrf_selected_ratios.update(
                    ucrf_log["ucrf_selected_ratio"]
                )
                ucrf_num_selecteds.update(
                    ucrf_log["ucrf_num_selected"]
                )
                ucrf_reliability_means.update(
                    ucrf_log["ucrf_reliability_mean"]
                )
                ucrf_selector_entropies.update(
                    ucrf_log["ucrf_selector_entropy"]
                )
                ucrf_selector_utility_agreements.update(
                    ucrf_log["ucrf_selector_utility_agreement"]
                )
                ucrf_unlabeled_utility_agreements.update(
                    ucrf_log["ucrf_unlabeled_utility_agreement"]
                )
                ucrf_text_candidate_accs.update(
                    ucrf_log["ucrf_text_candidate_acc"]
                )
                ucrf_image_candidate_accs.update(
                    ucrf_log["ucrf_image_candidate_acc"]
                )
                ucrf_common_candidate_accs.update(
                    ucrf_log["ucrf_common_candidate_acc"]
                )
                ucrf_oracle_candidate_accs.update(
                    ucrf_log["ucrf_oracle_candidate_acc"]
                )
                ucrf_selector_candidate_accs.update(
                    ucrf_log["ucrf_selector_candidate_acc"]
                )
                ucrf_selector_confidences.update(
                    float(
                        ucrf_selector_confidence_lb.mean().detach().cpu()
                    )
                )
                ucrf_selector_sum += np.asarray(
                    ucrf_log["ucrf_selector_mean"],
                    dtype=np.float64,
                )
                ucrf_stat_batches += 1
            total_losses.update(total_loss.cpu().detach())
            mask_ratios.update(mask.float().mean().cpu().detach())
            
            lr_last = self.optimizer.param_groups[0]['lr']
            
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(self.optimizer)
                if (
                    getattr(args, "use_msd", False)
                    or getattr(args, "use_dctr_msg", False)
                    or getattr(args, "use_ucrf", False)
                ):
                    selector_grad_sq = 0.0
                    for name, parameter in self.model.named_parameters():
                        if "modal_select_layer" in name and parameter.grad is not None:
                            selector_grad_sq += float(parameter.grad.detach().norm(2).cpu()) ** 2
                    selector_grad_norm = selector_grad_sq ** 0.5
                    if getattr(args, "use_msd", False):
                        msd_selector_grad_norms.update(selector_grad_norm)
                    if getattr(args, "use_dctr_msg", False):
                        dctr_msg_selector_grad_norms.update(
                            selector_grad_norm
                        )
                    if getattr(args, "use_ucrf", False):
                        ucrf_selector_grad_norms.update(
                            selector_grad_norm
                        )
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (
                    getattr(args, "use_msd", False)
                    or getattr(args, "use_dctr_msg", False)
                    or getattr(args, "use_ucrf", False)
                ):
                    selector_grad_sq = 0.0
                    for name, parameter in self.model.named_parameters():
                        if "modal_select_layer" in name and parameter.grad is not None:
                            selector_grad_sq += float(parameter.grad.detach().norm(2).cpu()) ** 2
                    selector_grad_norm = selector_grad_sq ** 0.5
                    if getattr(args, "use_msd", False):
                        msd_selector_grad_norms.update(selector_grad_norm)
                    if getattr(args, "use_dctr_msg", False):
                        dctr_msg_selector_grad_norms.update(
                            selector_grad_norm
                        )
                    if getattr(args, "use_ucrf", False):
                        ucrf_selector_grad_norms.update(
                            selector_grad_norm
                        )
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()
            batch_model_time.update(start_run.elapsed_time(end_run) / 1000)

            # self.it += 1
            start_batch.record()

        # self.print_fn(self.relation_matrix)

        self.print_fn(
            "[TRAIN-DIAG] epoch={} sup_loss={:.6f} unsup_loss={:.6f} "
            "decouple_loss={:.6f} total_loss={:.6f} mask_ratio={:.6f} "
            "pseudo_acc_oracle_for_debug_only={:.6f} "
            "pseudo_conf_mean={:.6f} pseudo_conf_std={:.6f} "
            "selected_conf_mean={:.6f} unselected_conf_mean={:.6f}".format(
                epoch,
                float(sup_losses.avg),
                float(unsup_losses.avg),
                float(decouple_losses.avg),
                float(total_losses.avg),
                diag_masked_count / max(diag_total_count, 1),
                diag_masked_correct / max(diag_masked_count, 1),
                float(np.mean(diag_confidences)) if diag_confidences else 0.0,
                float(np.std(diag_confidences)) if diag_confidences else 0.0,
                float(np.mean(diag_selected_confidences))
                if diag_selected_confidences
                else 0.0,
                float(np.mean(diag_unselected_confidences))
                if diag_unselected_confidences
                else 0.0,
            )
        )
        if getattr(args, "log_pseudo_diag", True):
            self.print_fn(
                "[PLF-DIAG] epoch={} candidate_coverage={:.6f} "
                "accepted_coverage={:.6f} weighted_coverage={:.6f} "
                "masked_pseudo_acc_oracle={:.6f} "
                "weighted_pseudo_acc_oracle={:.6f}".format(
                    epoch,
                    diag_candidate_count / max(diag_total_count, 1),
                    diag_masked_count / max(diag_total_count, 1),
                    diag_weight_sum / max(diag_total_count, 1),
                    diag_masked_correct / max(diag_masked_count, 1),
                    diag_weighted_correct / max(diag_weight_sum, 1e-12),
                )
            )
            self.print_fn(
                "[PSEUDO-DIAG] epoch={} all_acc={:.6f} masked_acc={:.6f} "
                "weighted_acc={:.6f} candidate_coverage={:.6f} "
                "accepted_coverage={:.6f} weighted_coverage={:.6f}".format(
                    epoch,
                    diag_all_correct / max(diag_total_count, 1),
                    diag_masked_correct / max(diag_masked_count, 1),
                    diag_weighted_correct / max(diag_weight_sum, 1e-12),
                    diag_candidate_count / max(diag_total_count, 1),
                    diag_masked_count / max(diag_total_count, 1),
                    diag_weight_sum / max(diag_total_count, 1),
                )
            )

        self.print_fn("Epoch {}/{} train: data time: {}, model time: {}, last lr: {}, labeled loss: {}, unlabeled loss: {}, decouple_loss: {}, total_loss: {}, mask ratio: {}, pseudo label correct ratio: {}".
                      format(epoch, args.epoch, batch_data_time.avg, batch_model_time.avg, lr_last, sup_losses.avg, unsup_losses.avg, decouple_losses.avg,total_losses.avg, mask_ratios.avg, pseudo_true_ratios.avg))
        if self.use_mllm_verification:
            pred_total = max(float(mllm_final_pred_counts.sum()), 1.0)
            final_pred_dist = (mllm_final_pred_counts / pred_total).round(6).tolist()
            weighted_pred_total = max(float(mllm_weighted_final_pred_sums.sum()), 1.0)
            weighted_final_pred_dist = (mllm_weighted_final_pred_sums / weighted_pred_total).round(6).tolist()
            self.print_fn(
                "Epoch {}/{} MLLM verification: mode: {}, selective: {}, policy: {}, action: {}, base_coverage: {}, final_coverage: {}, weighted_coverage: {}, retention: {}, base_pseudo_acc: {}, final_pseudo_acc: {}, weighted_pseudo_acc: {}, effective_signal: {}, reject_error_rate: {}, missing_ratio: {}, mm_agree_ratio: {}, internal_conflict_ratio: {}, external_conflict_ratio: {}, high_risk_ratio: {}, mllm_used_ratio: {}, mllm_agree_ratio_on_risk: {}, mllm_reject_ratio_on_risk: {}, low_risk_pseudo_acc: {}, high_risk_base_pseudo_acc: {}, high_risk_final_pseudo_acc: {}, ecs_rescued_coverage: {}, ecs_rescued_pseudo_acc: {}, ecs_opposite_conflict_pseudo_acc: {}, ecs_emotional_consensus_count: {}, ecs_neutral_consensus_count: {}, ecs_modality_imbalance_count: {}, ecs_opposite_conflict_count: {}, ecs_qwen_agree_pass_count: {}, ecs_qwen_disagree_pass_count: {}, hybrid_support_mean: {}, hybrid_q_conf_mean: {}, hybrid_q_margin_mean: {}, hybrid_strong_support_ratio: {}, hybrid_uncertain_ratio: {}, hybrid_high_conflict_ratio: {}, hybrid_opposite_conflict_ratio: {}, hybrid_neutral_limited_ratio: {}, final_pred_dist: {}, weighted_final_pred_dist: {}".format(
                    epoch,
                    args.epoch,
                    self.mllm_verifier.mode,
                    getattr(args, "mllm_selective_verify", True),
                    getattr(args, "mllm_verify_policy", "risky_only"),
                    getattr(args, "mllm_action", "veto"),
                    mllm_base_coverages.avg,
                    mllm_final_coverages.avg,
                    mllm_weighted_coverages.avg,
                    mllm_retentions.avg,
                    mllm_base_pseudo_accs.avg,
                    mllm_final_pseudo_accs.avg,
                    mllm_weighted_pseudo_accs.avg,
                    mllm_effective_signals.avg,
                    mllm_reject_error_rates.avg,
                    mllm_missing_ratios.avg,
                    mllm_mm_agree_ratios.avg,
                    mllm_internal_conflict_ratios.avg,
                    mllm_external_conflict_ratios.avg,
                    mllm_high_risk_ratios.avg,
                    mllm_used_ratios.avg,
                    mllm_agree_on_risk_ratios.avg,
                    mllm_reject_on_risk_ratios.avg,
                    mllm_low_risk_pseudo_accs.avg,
                    mllm_high_risk_base_pseudo_accs.avg,
                    mllm_high_risk_final_pseudo_accs.avg,
                    mllm_ecs_rescued_coverages.avg,
                    mllm_ecs_rescued_pseudo_accs.avg,
                    mllm_ecs_opposite_conflict_pseudo_accs.avg,
                    mllm_ecs_emotional_consensus_counts.avg,
                    mllm_ecs_neutral_consensus_counts.avg,
                    mllm_ecs_modality_imbalance_counts.avg,
                    mllm_ecs_opposite_conflict_counts.avg,
                    mllm_ecs_qwen_agree_pass_counts.avg,
                    mllm_ecs_qwen_disagree_pass_counts.avg,
                    mllm_hybrid_support_means.avg,
                    mllm_hybrid_q_conf_means.avg,
                    mllm_hybrid_q_margin_means.avg,
                    mllm_hybrid_strong_support_ratios.avg,
                    mllm_hybrid_uncertain_ratios.avg,
                    mllm_hybrid_high_conflict_ratios.avg,
                    mllm_hybrid_opposite_conflict_ratios.avg,
                    mllm_hybrid_neutral_limited_ratios.avg,
                    final_pred_dist,
                    weighted_final_pred_dist,
                )
            )

        if (
            self.use_mllm_verification
            and getattr(args, "mllm_action", "veto") == "dctr_plf"
        ):
            self.print_fn(
                "Epoch {}/{} DCTR-PLF: prior_mean: {}, prior_strength: {}, "
                "min_reliability: {}, apply_all: {}, "
                "labeled_external_seen: {}, labeled_internal_seen_epoch: {}, "
                "reliability(m/t/v): [{}, {}, {}], "
                "external_precision(m/t/v): [{}, {}, {}], "
                "internal_precision(m/t/v): [{}, {}, {}], "
                "aug_stability(m/t/v): [{}, {}, {}], "
                "distribution_similarity(m/t/v): [{}, {}, {}], "
                "weighted_coverage(m/t/v): [{}, {}, {}], "
                "weighted_pseudo_acc_oracle(m/t/v): [{}, {}, {}]".format(
                    epoch,
                    args.epoch,
                    getattr(args, "dctr_prior_mean", 0.5),
                    getattr(args, "dctr_prior_strength", 6.0),
                    getattr(args, "dctr_min_reliability", 0.05),
                    getattr(args, "dctr_apply_all", True),
                    len(self.dctr_external_seen),
                    len(self.dctr_internal_seen_epoch),
                    dctr_reliability_m.avg,
                    dctr_reliability_t.avg,
                    dctr_reliability_v.avg,
                    dctr_external_precision_m.avg,
                    dctr_external_precision_t.avg,
                    dctr_external_precision_v.avg,
                    dctr_internal_precision_m.avg,
                    dctr_internal_precision_t.avg,
                    dctr_internal_precision_v.avg,
                    dctr_aug_stability_m.avg,
                    dctr_aug_stability_t.avg,
                    dctr_aug_stability_v.avg,
                    dctr_distribution_similarity_m.avg,
                    dctr_distribution_similarity_t.avg,
                    dctr_distribution_similarity_v.avg,
                    dctr_weighted_coverage_m.avg,
                    dctr_weighted_coverage_t.avg,
                    dctr_weighted_coverage_v.avg,
                    dctr_weighted_pseudo_acc_m.avg,
                    dctr_weighted_pseudo_acc_t.avg,
                    dctr_weighted_pseudo_acc_v.avg,
                )
            )

        if getattr(args, "use_ce_umc", False):
            self.print_fn(
                "Epoch {}/{} CE-UMC: scope: {}, evidence_shuffle: {}, "
                "lambda_text: {}, lambda_image: {}, "
                "effective_lambda_text: {}, effective_lambda_image: {}, "
                "warmup_epoch: {}, rampup_epoch: {}, target_temperature: {}, "
                "text_conf_threshold: {}, image_conf_threshold: {}, "
                "min_certainty: {}, loss_text: {}, loss_image: {}, "
                "scope_ratio: {}, text_selected_ratio: {}, "
                "image_selected_ratio: {}, text_num_selected: {}, "
                "image_num_selected: {}, text_confidence_mean: {}, "
                "image_confidence_mean: {}, text_certainty_mean: {}, "
                "image_certainty_mean: {}, text_weight_mean: {}, "
                "image_weight_mean: {}, text_model_evidence_agreement: {}, "
                "image_model_evidence_agreement: {}".format(
                    epoch,
                    args.epoch,
                    getattr(args, "ce_umc_scope", "all"),
                    getattr(args, "ce_umc_evidence_shuffle", False),
                    getattr(args, "lambda_ce_umc_text", 0.02),
                    getattr(args, "lambda_ce_umc_image", 0.05),
                    ce_umc_text_effective_lambdas.avg,
                    ce_umc_image_effective_lambdas.avg,
                    getattr(args, "ce_umc_warmup_epoch", 10),
                    getattr(args, "ce_umc_rampup_epoch", 20),
                    getattr(args, "ce_umc_target_temperature", 2.0),
                    getattr(args, "ce_umc_text_conf_threshold", 0.90),
                    getattr(args, "ce_umc_image_conf_threshold", 0.90),
                    getattr(args, "ce_umc_min_certainty", 0.20),
                    ce_umc_text_losses.avg,
                    ce_umc_image_losses.avg,
                    ce_umc_scope_ratios.avg,
                    ce_umc_text_selected_ratios.avg,
                    ce_umc_image_selected_ratios.avg,
                    ce_umc_text_num_selecteds.avg,
                    ce_umc_image_num_selecteds.avg,
                    ce_umc_text_confidence_means.avg,
                    ce_umc_image_confidence_means.avg,
                    ce_umc_text_certainty_means.avg,
                    ce_umc_image_certainty_means.avg,
                    ce_umc_text_weight_means.avg,
                    ce_umc_image_weight_means.avg,
                    ce_umc_text_agreements.avg,
                    ce_umc_image_agreements.avg,
                )
            )

        if getattr(args, "use_ec_pfd", False):
            self.print_fn(
                "Epoch {}/{} EC-PFD: lambda: {}, threshold: {}, use_soft_label: {}, loss_ec_pfd: {}, mask_ratio: {}, num_selected: {}, reliability_mean: {}, external_target_ratio: {}, neutral_uncertain_ratio: {}".format(
                    epoch,
                    args.epoch,
                    getattr(args, "lambda_ec_pfd", 0.2),
                    getattr(args, "ec_pfd_threshold", 0.75),
                    getattr(args, "ec_pfd_use_soft_label", True),
                    ec_pfd_losses.avg,
                    ec_pfd_mask_ratios.avg,
                    ec_pfd_num_selecteds.avg,
                    ec_pfd_reliability_means.avg,
                    ec_pfd_external_target_ratios.avg,
                    ec_pfd_neutral_uncertain_ratios.avg,
                )
            )

        if getattr(args, "use_ead_a", False):
            self.print_fn(
                "Epoch {}/{} EAD-A: lambda: {}, effective_lambda: {}, "
                "warmup_epoch: {}, reliability_threshold: {}, evidence_shuffle: {}, "
                "loss: {}, text_anchor_loss: {}, image_anchor_loss: {}, "
                "common_align_loss: {}, selected_ratio: {}, num_selected: {}, "
                "reliability_mean: {}, text_conf_mean: {}, image_conf_mean: {}, "
                "agreement_mean: {}, opposite_ratio: {}, neutral_ratio: {}, "
                "text_weight_mean: {}, image_weight_mean: {}, common_weight_mean: {}".format(
                    epoch,
                    args.epoch,
                    getattr(args, "lambda_ead_a", 0.02),
                    ead_a_effective_lambdas.avg,
                    getattr(args, "ead_a_warmup_epoch", 30),
                    getattr(args, "ead_a_reliability_threshold", 0.45),
                    getattr(args, "ead_a_evidence_shuffle", False),
                    ead_a_losses.avg,
                    ead_a_text_anchor_losses.avg,
                    ead_a_image_anchor_losses.avg,
                    ead_a_common_align_losses.avg,
                    ead_a_selected_ratios.avg,
                    ead_a_num_selecteds.avg,
                    ead_a_reliability_means.avg,
                    ead_a_text_conf_means.avg,
                    ead_a_image_conf_means.avg,
                    ead_a_agreement_means.avg,
                    ead_a_opposite_ratios.avg,
                    ead_a_neutral_ratios.avg,
                    ead_a_text_weight_means.avg,
                    ead_a_image_weight_means.avg,
                    ead_a_common_weight_means.avg,
                )
            )

        if getattr(args, "use_sa_dd", False):
            class_mass_total = float(sa_dd_class_weight_mass.sum())
            if class_mass_total > 0.0:
                sa_dd_class_mass_dist = (
                    sa_dd_class_weight_mass / class_mass_total
                ).round(6).tolist()
            else:
                sa_dd_class_mass_dist = [
                    0.0 for _ in range(self.num_classes)
                ]
            self.print_fn(
                "Epoch {}/{} SA-DD: version: {}, lambda: {}, effective_lambda: {}, "
                "warmup_epoch: {}, rampup_epoch: {}, ablation: {}, "
                "fixed_gate_value: {}, evidence_shuffle: {}, "
                "legacy_replace_unlabeled_sim: {}, "
                "loss: {}, common_loss: {}, private_loss: {}, selected_ratio: {}, "
                "num_selected: {}, gate_mean: {}, history_coverage: {}, "
                "history_stability_mean: {}, aug_stability_mean: {}, "
                "relation_agreement_mean: {}, relation_conf_mean: {}, "
                "stability_gate_mean: {}, model_uncertainty_mean: {}, "
                "ecplf_risk_mean: {}, information_mean: {}, "
                "relation_strength_mean: {}, same_score_mean: {}, "
                "opposite_score_mean: {}, uncertain_score_mean: {}, "
                "common_ratio_mean: {}, private_ratio_mean: {}, "
                "selected_accept_ratio: {}, selected_reject_ratio: {}, "
                "opposite_ratio: {}, neutral_ratio: {}, collapse_backoff_ratio: {}, "
                "epoch_pred_dist: {}, ema_pred_dist: {}, "
                "class_weight_mass_dist: {}".format(
                    epoch,
                    args.epoch,
                    getattr(args, "sa_dd_version", "v2"),
                    getattr(args, "lambda_sa_dd", 0.01),
                    sa_dd_effective_lambdas.avg,
                    getattr(args, "sa_dd_warmup_epoch", 30),
                    getattr(args, "sa_dd_rampup_epoch", 20),
                    getattr(args, "sa_dd_ablation", "none"),
                    getattr(args, "sa_dd_fixed_gate_value", 0.5),
                    getattr(args, "sa_dd_evidence_shuffle", False),
                    getattr(args, "sa_dd_replace_unlabeled_sim", True),
                    sa_dd_losses.avg,
                    sa_dd_common_losses.avg,
                    sa_dd_private_losses.avg,
                    sa_dd_selected_ratios.avg,
                    sa_dd_num_selecteds.avg,
                    sa_dd_gate_means.avg,
                    sa_dd_history_coverages.avg,
                    sa_dd_history_stability_means.avg,
                    sa_dd_aug_stability_means.avg,
                    sa_dd_relation_agreement_means.avg,
                    sa_dd_relation_conf_means.avg,
                    sa_dd_stability_gate_means.avg,
                    sa_dd_model_uncertainty_means.avg,
                    sa_dd_ecplf_risk_means.avg,
                    sa_dd_information_means.avg,
                    sa_dd_relation_strength_means.avg,
                    sa_dd_same_score_means.avg,
                    sa_dd_opposite_score_means.avg,
                    sa_dd_uncertain_score_means.avg,
                    sa_dd_common_ratio_means.avg,
                    sa_dd_private_ratio_means.avg,
                    sa_dd_selected_accept_ratios.avg,
                    sa_dd_selected_reject_ratios.avg,
                    sa_dd_opposite_ratios.avg,
                    sa_dd_neutral_ratios.avg,
                    sa_dd_collapse_backoffs.avg,
                    [round(value, 6) for value in sa_dd_epoch_pred_dist],
                    [round(value, 6) for value in sa_dd_ema_pred_dist],
                    sa_dd_class_mass_dist,
                )
            )

        if getattr(args, "use_msd", False):
            msd_divisor = max(msd_stat_batches, 1)
            msd_selector_mean = (msd_selector_sum / msd_divisor).round(6).tolist()
            msd_prior_mean = (msd_prior_sum / msd_divisor).round(6).tolist()
            self.print_fn(
                "Epoch {}/{} MSD: soft_selector: {}, selector_temperature: {}, "
                "lambda: {}, effective_lambda: {}, warmup_epoch: {}, rampup_epoch: {}, "
                "reliability_threshold: {}, prior_mode: {}, evidence_shuffle: {}, "
                "use_labeled_anchor: {}, use_unlabeled_evidence: {}, loss: {}, labeled_loss: {}, "
                "unlabeled_loss: {}, selected_ratio: {}, num_selected: {}, "
                "reliability_mean: {}, valid_evidence_ratio: {}, opposite_conflict_ratio: {}, "
                "neutral_consensus_ratio: {}, text_branch_acc: {}, image_branch_acc: {}, "
                "multimodal_branch_acc: {}, oracle_branch_acc: {}, selector_branch_acc: {}, "
                "prior_selected_branch_acc: {}, "
                "selector_mean[text,image,multimodal]: {}, prior_mean[text,image,multimodal]: {}, "
                "selector_grad_norm: {}".format(
                    epoch,
                    args.epoch,
                    getattr(args, "use_soft_modal_selector", False),
                    getattr(args, "modal_selector_temperature", 1.0),
                    getattr(args, "lambda_msd", 0.1),
                    msd_effective_lambdas.avg,
                    getattr(args, "msd_warmup_epoch", 20),
                    getattr(args, "msd_rampup_epoch", 20),
                    getattr(args, "msd_reliability_threshold", 0.75),
                    getattr(args, "msd_prior_mode", "evidence"),
                    getattr(args, "msd_evidence_shuffle", False),
                    getattr(args, "msd_use_labeled_anchor", True),
                    getattr(args, "msd_use_unlabeled_evidence", True),
                    msd_losses.avg,
                    msd_labeled_losses.avg,
                    msd_unlabeled_losses.avg,
                    msd_selected_ratios.avg,
                    msd_num_selecteds.avg,
                    msd_reliability_means.avg,
                    msd_valid_evidence_ratios.avg,
                    msd_opposite_conflict_ratios.avg,
                    msd_neutral_consensus_ratios.avg,
                    msd_text_branch_accs.avg,
                    msd_image_branch_accs.avg,
                    msd_multimodal_branch_accs.avg,
                    msd_oracle_branch_accs.avg,
                    msd_selector_branch_accs.avg,
                    msd_prior_selected_branch_accs.avg,
                    msd_selector_mean,
                    msd_prior_mean,
                    msd_selector_grad_norms.avg,
                )
            )

        if getattr(args, "use_dctr_msg", False):
            dctr_msg_divisor = max(dctr_msg_stat_batches, 1)
            dctr_msg_selector_mean = (
                dctr_msg_selector_sum / dctr_msg_divisor
            ).round(6).tolist()
            dctr_msg_prior_mean = (
                dctr_msg_prior_sum / dctr_msg_divisor
            ).round(6).tolist()
            self.print_fn(
                "Epoch {}/{} DCTR-MSG: soft_selector: {}, "
                "selector_temperature: {}, lambda: {}, effective_lambda: {}, "
                "warmup_epoch: {}, rampup_epoch: {}, "
                "reliability_threshold: {}, prior_temperature: {}, "
                "min_margin: {}, exclude_opposite: {}, "
                "reliability_shuffle: {}, "
                "loss: {}, labeled_loss: {}, unlabeled_loss: {}, "
                "selected_ratio: {}, num_selected: {}, reliability_mean: {}, "
                "reliability_margin_mean: {}, selector_entropy: {}, "
                "prior_entropy: {}, selector_prior_agreement: {}, "
                "branch_acc[text,image,multimodal]: [{}, {}, {}], "
                "oracle_branch_acc: {}, selector_branch_acc: {}, "
                "prior_branch_acc: {}, "
                "selector_mean[text,image,multimodal]: {}, "
                "prior_mean[text,image,multimodal]: {}, "
                "selector_grad_norm: {}".format(
                    epoch,
                    args.epoch,
                    getattr(args, "use_soft_modal_selector", False),
                    getattr(args, "modal_selector_temperature", 1.0),
                    getattr(args, "lambda_dctr_msg", 0.03),
                    dctr_msg_effective_lambdas.avg,
                    getattr(args, "dctr_msg_warmup_epoch", 20),
                    getattr(args, "dctr_msg_rampup_epoch", 20),
                    getattr(
                        args,
                        "dctr_msg_reliability_threshold",
                        0.45,
                    ),
                    getattr(args, "dctr_msg_prior_temperature", 0.5),
                    getattr(args, "dctr_msg_min_margin", 0.02),
                    getattr(
                        args,
                        "dctr_msg_exclude_opposite_conflict",
                        False,
                    ),
                    getattr(
                        args,
                        "dctr_msg_reliability_shuffle",
                        False,
                    ),
                    dctr_msg_losses.avg,
                    dctr_msg_labeled_losses.avg,
                    dctr_msg_unlabeled_losses.avg,
                    dctr_msg_selected_ratios.avg,
                    dctr_msg_num_selecteds.avg,
                    dctr_msg_reliability_means.avg,
                    dctr_msg_reliability_margin_means.avg,
                    dctr_msg_selector_entropies.avg,
                    dctr_msg_prior_entropies.avg,
                    dctr_msg_selector_prior_agreements.avg,
                    dctr_msg_text_branch_accs.avg,
                    dctr_msg_image_branch_accs.avg,
                    dctr_msg_multimodal_branch_accs.avg,
                    dctr_msg_oracle_branch_accs.avg,
                    dctr_msg_selector_branch_accs.avg,
                    dctr_msg_prior_branch_accs.avg,
                    dctr_msg_selector_mean,
                    dctr_msg_prior_mean,
                    dctr_msg_selector_grad_norms.avg,
                )
            )

        if getattr(args, "use_ucrf", False):
            ucrf_divisor = max(ucrf_stat_batches, 1)
            ucrf_selector_mean = (
                ucrf_selector_sum / ucrf_divisor
            ).round(6).tolist()
            self.print_fn(
                "Epoch {}/{} UCRF: lambda: {}, warmup_epoch: {}, "
                "rampup_epoch: {}, residual_beta_max: {}, "
                "effective_beta: {}, unlabeled_scale: {}, "
                "utility_temperature: {}, reliability_threshold: {}, "
                "loss: {}, labeled_loss: {}, unlabeled_loss: {}, "
                "selected_ratio: {}, num_selected: {}, "
                "reliability_mean: {}, selector_entropy: {}, "
                "selector_confidence: {}, "
                "selector_utility_agreement: {}, "
                "unlabeled_utility_agreement: {}, "
                "candidate_acc[text,image,common]: [{}, {}, {}], "
                "oracle_candidate_acc: {}, selector_candidate_acc: {}, "
                "selector_mean[text,image,common]: {}, "
                "selector_grad_norm: {}".format(
                    epoch,
                    args.epoch,
                    getattr(args, "lambda_ucrf", 0.05),
                    getattr(args, "ucrf_warmup_epoch", 20),
                    getattr(args, "ucrf_rampup_epoch", 20),
                    getattr(args, "ucrf_residual_beta", 0.3),
                    ucrf_effective_beta,
                    ucrf_unlabeled_scale,
                    getattr(args, "ucrf_utility_temperature", 0.5),
                    getattr(args, "ucrf_reliability_threshold", 0.75),
                    ucrf_losses.avg,
                    ucrf_labeled_losses.avg,
                    ucrf_unlabeled_losses.avg,
                    ucrf_selected_ratios.avg,
                    ucrf_num_selecteds.avg,
                    ucrf_reliability_means.avg,
                    ucrf_selector_entropies.avg,
                    ucrf_selector_confidences.avg,
                    ucrf_selector_utility_agreements.avg,
                    ucrf_unlabeled_utility_agreements.avg,
                    ucrf_text_candidate_accs.avg,
                    ucrf_image_candidate_accs.avg,
                    ucrf_common_candidate_accs.avg,
                    ucrf_oracle_candidate_accs.avg,
                    ucrf_selector_candidate_accs.avg,
                    ucrf_selector_mean,
                    ucrf_selector_grad_norms.avg,
                )
            )

        eval_split_name = (
            "val" if getattr(args, "val_data_dir", None) else "test"
        )
        eval_dict = self.evaluate(
            args=args,
            epoch=epoch,
            split_name=eval_split_name,
        )
        is_best = eval_dict['eval/top-1-acc'] > best_eval_acc
        best_eval_acc = max(best_eval_acc, eval_dict['eval/top-1-acc'])
        class_f1 = eval_dict['eval/class-f1']
        self.print_fn(
            "Epoch {}/{} {}: loss: {:.6f}, Accuracy: {:.6f}, "
            "Macro-F1: {:.6f}, Positive-F1: {:.6f}, Negative-F1: {:.6f}, "
            "Neutral-F1: {:.6f}, best Accuracy: {:.6f}".format(
                epoch,
                args.epoch,
                eval_split_name,
                eval_dict['eval/loss'],
                eval_dict['eval/top-1-acc'],
                eval_dict['eval/macro-f1'],
                class_f1[0],
                class_f1[1],
                class_f1[2],
                best_eval_acc,
            )
        )

        save_path = os.path.join(args.save_dir, args.save_name)
        if is_best:
            self.best_epoch = epoch
            self.save_model('model_best.pth', save_path)
            if getattr(args, "export_best_epoch_analysis", True):
                self._save_evaluation_analysis(
                    eval_dict=eval_dict,
                    save_path=save_path,
                    file_prefix="best_epoch",
                    epoch=epoch,
                    split_name=eval_split_name,
                )
        return eval_dict['eval/top-1-acc']

    @torch.no_grad()
    def dump_role_reliability_diag(self, diag_loader, args, output_path):
        """Dump sample-level HH/HL/LH/LL diagnostics for role-dependent reliability."""

        def _to_text_list(items):
            out = []
            for item in items:
                out.append(item.item() if hasattr(item, "item") else item)
            return out

        def _ids_to_list(sample_ids):
            if torch.is_tensor(sample_ids):
                return [
                    str(value)
                    for value in sample_ids.detach().cpu().view(-1).tolist()
                ]
            return [
                str(value.item() if hasattr(value, "item") else value)
                for value in sample_ids
            ]

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_path = output_path + ".summary.json"

        self.model.eval()
        counters = Counter()
        total = 0
        pseudo_correct_total = 0
        rad_selected_total = 0
        ecr_accept_total = 0
        ecr_weight_sum = 0.0

        with open(output_path, "w", encoding="utf-8") as fout:
            for batch in tqdm(diag_loader, desc="role-reliability diagnostic"):
                (
                    x_ulb_idx,
                    x_ulb_w,
                    x_ulb_s0,
                    x_ulb_s1,
                    t_ulb,
                    y_ulb,
                ) = batch

                num_ulb = x_ulb_w.shape[0]
                x_ulb_w = x_ulb_w.cuda(args.gpu)
                x_ulb_s0 = x_ulb_s0.cuda(args.gpu)
                x_ulb_s1 = x_ulb_s1.cuda(args.gpu)
                y_ulb = y_ulb.cuda(args.gpu)

                img_inputs = torch.cat((x_ulb_w, x_ulb_s0, x_ulb_s1))
                t_ulb_list = _to_text_list(t_ulb)
                text_input = t_ulb_list + t_ulb_list + t_ulb_list
                text_input = self.tokenizer(
                    text_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                text_input = {
                    key: value.cuda(args.gpu)
                    for key, value in text_input.items()
                }

                output = self.model(img_inputs, text_input)
                logits_m = output["pre_m_att"]
                logits_t = output["pre_t"]
                logits_v = output["pre_v"]
                logits_x_ulb_w, logits_x_ulb_s0, _ = torch.split(logits_m, num_ulb)
                logits_t_ulb_w, _, _ = torch.split(logits_t, num_ulb)
                logits_v_ulb_w, _, _ = torch.split(logits_v, num_ulb)

                ulb_probs = torch.softmax(logits_x_ulb_w.detach(), dim=1)
                t_ulb_probs = torch.softmax(logits_t_ulb_w.detach(), dim=1)
                v_ulb_probs = torch.softmax(logits_v_ulb_w.detach(), dim=1)
                scores, lbs_u_guess = torch.max(ulb_probs, dim=1)
                _, t_lbs_u_guess = torch.max(t_ulb_probs, dim=1)
                _, v_lbs_u_guess = torch.max(v_ulb_probs, dim=1)

                mask = scores.ge(args.threshold)
                mllm_sample_weights = torch.ones_like(scores, dtype=torch.float)

                if self.use_mllm_verification:
                    branch_conflict = (
                        t_lbs_u_guess.ne(v_lbs_u_guess)
                        | t_lbs_u_guess.ne(lbs_u_guess)
                        | v_lbs_u_guess.ne(lbs_u_guess)
                    )
                    top2_scores = torch.topk(ulb_probs, k=2, dim=1).values
                    margin = top2_scores[:, 0] - top2_scores[:, 1]
                    low_margin = margin.lt(getattr(args, "risk_margin_threshold", 0.2))
                    strong_probs_for_risk = torch.softmax(
                        logits_x_ulb_s0.detach(),
                        dim=1,
                    )
                    weak_strong_kl = F.kl_div(
                        torch.log(strong_probs_for_risk.clamp_min(1e-12)),
                        ulb_probs,
                        reduction="none",
                    ).sum(dim=1)
                    unstable_aug = weak_strong_kl.gt(
                        getattr(args, "risk_kl_threshold", 0.5)
                    )
                    pos_id = self.mllm_verifier.label_map["positive"]
                    neg_id = self.mllm_verifier.label_map["negative"]
                    modality_conflict = (
                        (t_lbs_u_guess.eq(pos_id) & v_lbs_u_guess.eq(neg_id))
                        | (t_lbs_u_guess.eq(neg_id) & v_lbs_u_guess.eq(pos_id))
                    )
                    high_risk = (
                        branch_conflict
                        | low_margin
                        | unstable_aug
                        | modality_conflict
                    )
                    if getattr(args, "mllm_selective_verify", True):
                        verify_mask = (
                            torch.ones_like(mask, dtype=torch.bool)
                            if getattr(args, "mllm_verify_policy", "risky_only") == "all"
                            else high_risk
                        )
                    else:
                        verify_mask = torch.ones_like(mask, dtype=torch.bool)

                    mask, _, mllm_sample_weights, override_labels = (
                        self.mllm_verifier.verify(
                            base_mask=mask,
                            sample_ids=x_ulb_idx,
                            y_t=t_lbs_u_guess,
                            y_v=v_lbs_u_guess,
                            y_m=lbs_u_guess,
                            y_true=y_ulb,
                            verify_mask=verify_mask,
                            action=getattr(args, "mllm_action", "veto"),
                            soft_weight=getattr(args, "mllm_soft_weight", 0.5),
                            ecs_conf_threshold=getattr(
                                args,
                                "mllm_ecs_conf_threshold",
                                0.75,
                            ),
                            hybrid_config={
                                "support_high": getattr(args, "hybrid_support_high", 0.65),
                                "support_low": getattr(args, "hybrid_support_low", 0.30),
                                "qwen_conf_high": getattr(args, "hybrid_qwen_conf_high", 0.75),
                                "weight_agree": getattr(args, "hybrid_weight_agree", 1.0),
                                "weight_uncertain": getattr(args, "hybrid_weight_uncertain", 0.5),
                                "weight_high_conflict": getattr(args, "hybrid_weight_high_conflict", 0.0),
                                "weight_opposite_conflict": getattr(args, "hybrid_weight_opposite_conflict", 0.0),
                                "neutral_max_weight": getattr(args, "hybrid_neutral_max_weight", 0.5),
                                "use_distribution": getattr(args, "hybrid_use_distribution", True),
                                "use_opposite_conflict": getattr(args, "hybrid_use_opposite_conflict", True),
                            },
                        )
                    )
                    if override_labels is not None:
                        lbs_u_guess = override_labels.detach()

                history_stability, _ = self._sa_dd_history_stability(
                    sample_ids=x_ulb_idx,
                    current_prob=ulb_probs,
                    momentum=getattr(args, "sa_dd_history_momentum", 0.9),
                )
                qwen_sa_dd = self.mllm_verifier.msd_evidence(
                    sample_ids=x_ulb_idx,
                    device=lbs_u_guess.device,
                )
                if getattr(args, "sa_dd_evidence_shuffle", False):
                    permutation = torch.randperm(num_ulb, device=lbs_u_guess.device)
                    qwen_sa_dd = {
                        key: value[permutation]
                        for key, value in qwen_sa_dd.items()
                    }

                common_text_s0 = torch.split(output["c_l"], num_ulb)[1]
                common_image_s0 = torch.split(output["c_v"], num_ulb)[1]
                private_text_s0 = torch.split(output["s_l"], num_ulb)[1]
                private_image_s0 = torch.split(output["s_v"], num_ulb)[1]
                strong_prob = torch.softmax(logits_x_ulb_s0.detach(), dim=1)
                _, _, rad_selected = compute_sa_dd_loss(
                    common_text=common_text_s0,
                    common_image=common_image_s0,
                    private_text=private_text_s0,
                    private_image=private_image_s0,
                    weak_prob=ulb_probs,
                    strong_prob=strong_prob,
                    qwen_evidence=qwen_sa_dd,
                    history_stability=history_stability,
                    pseudo_label=lbs_u_guess.detach(),
                    ecplf_accept=mask.detach(),
                    ecplf_weight=mllm_sample_weights.detach(),
                    num_classes=self.num_classes,
                    min_history_stability=getattr(args, "sa_dd_min_history_stability", 0.6),
                    min_aug_stability=getattr(args, "sa_dd_min_aug_stability", 0.6),
                    min_relation_conf=getattr(args, "sa_dd_min_relation_conf", 0.2),
                    relation_floor=getattr(args, "sa_dd_relation_floor", 0.2),
                    relation_ceiling=getattr(args, "sa_dd_relation_ceiling", 0.8),
                    common_weight=getattr(args, "sa_dd_common_weight", 1.0),
                    private_weight=getattr(args, "sa_dd_private_weight", 1.0),
                    ecplf_floor=getattr(args, "sa_dd_ecplf_floor", 0.5),
                    neutral_scale=getattr(args, "sa_dd_neutral_scale", 0.5),
                    min_samples=getattr(args, "sa_dd_min_samples", 2),
                    collapse_backoff=False,
                    version=getattr(args, "sa_dd_version", "v2"),
                    positive_id=self.mllm_verifier.positive_id,
                    negative_id=self.mllm_verifier.negative_id,
                    v2_min_gate=getattr(args, "sa_dd_v2_min_gate", 0.05),
                    v2_stability_temperature=getattr(
                        args,
                        "sa_dd_v2_stability_temperature",
                        0.1,
                    ),
                    ablation=getattr(args, "sa_dd_ablation", "none"),
                    fixed_gate_value=getattr(args, "sa_dd_fixed_gate_value", 0.5),
                )

                sample_ids = _ids_to_list(x_ulb_idx)
                pseudo_correct = lbs_u_guess.eq(y_ulb)
                for offset, sample_id in enumerate(sample_ids):
                    sup_high = bool(pseudo_correct[offset].detach().cpu())
                    rel_high = bool(rad_selected[offset].detach().cpu())
                    if sup_high and rel_high:
                        group = "HH"
                    elif sup_high and not rel_high:
                        group = "HL"
                    elif (not sup_high) and rel_high:
                        group = "LH"
                    else:
                        group = "LL"

                    row = {
                        "id": sample_id,
                        "true_label": int(y_ulb[offset].detach().cpu()),
                        "pseudo_label": int(lbs_u_guess[offset].detach().cpu()),
                        "pseudo_conf": float(scores[offset].detach().cpu()),
                        "pseudo_correct": int(sup_high),
                        "ecr_accept": int(bool(mask[offset].detach().cpu())),
                        "ecr_weight": float(mllm_sample_weights[offset].detach().cpu()),
                        "rad_selected": int(rel_high),
                        "group": group,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

                    counters[group] += 1
                    total += 1
                    pseudo_correct_total += int(sup_high)
                    rad_selected_total += int(rel_high)
                    ecr_accept_total += int(bool(mask[offset].detach().cpu()))
                    ecr_weight_sum += float(mllm_sample_weights[offset].detach().cpu())

        summary = {
            "total": total,
            "counts": {
                key: int(counters.get(key, 0))
                for key in ["HH", "HL", "LH", "LL"]
            },
            "percent": {
                key: (100.0 * float(counters.get(key, 0)) / float(total))
                if total
                else 0.0
                for key in ["HH", "HL", "LH", "LL"]
            },
            "pseudo_correct_ratio": float(pseudo_correct_total) / float(total) if total else 0.0,
            "rad_selected_ratio": float(rad_selected_total) / float(total) if total else 0.0,
            "ecr_accept_ratio": float(ecr_accept_total) / float(total) if total else 0.0,
            "ecr_weight_mean": float(ecr_weight_sum) / float(total) if total else 0.0,
            "definition": {
                "HH": "pseudo-label correct and selected by RAD",
                "HL": "pseudo-label correct but not selected by RAD",
                "LH": "pseudo-label incorrect but selected by RAD",
                "LL": "pseudo-label incorrect and not selected by RAD",
            },
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        self.print_fn(f"[ROLE-DIAG] jsonl={output_path}")
        self.print_fn(f"[ROLE-DIAG] summary={summary_path}")
        self.print_fn(f"[ROLE-DIAG] counts={summary['counts']} percent={summary['percent']}")

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, epoch=None, split_name=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        sample_records = []
        eval_split = split_name or (
            "val" if getattr(args, "val_data_dir", None) else "test"
        )

        for sample_ids, x, text_input, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            raw_text = [
                item.item() if hasattr(item, "item") else str(item)
                for item in text_input
            ]
            sample_id_list = [
                item.item() if hasattr(item, "item") else str(item)
                for item in sample_ids
            ]
            text_input = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True)
            text_input = {key: value.cuda(args.gpu) for key, value in text_input.items()}
            num_batch = x.shape[0]
            total_num += num_batch
            output = self.model(x, text_input)
            # logits = output['pre_m']
            logits = output['pre_m_att']
            loss = F.cross_entropy(logits, y, reduction='mean')
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            true_cpu = y.cpu().tolist()
            pred_cpu = pred.cpu().detach().tolist()
            prob_cpu = probs.cpu().detach().tolist()
            conf_cpu = conf.cpu().detach().tolist()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred_cpu)
            y_logits.extend(prob_cpu)
            total_loss += loss.cpu().detach() * num_batch
            for sample_id, text, true_label, pred_label, conf_value, prob in zip(
                sample_id_list,
                raw_text,
                true_cpu,
                pred_cpu,
                conf_cpu,
                prob_cpu,
            ):
                sample_records.append(
                    {
                        "sample_id": str(sample_id),
                        "text": text,
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "true_class": self._label_name(int(true_label)),
                        "pred_class": self._label_name(int(pred_label)),
                        "correct": bool(int(true_label) == int(pred_label)),
                        "confidence": float(conf_value),
                        "prob_positive": float(prob[0]),
                        "prob_negative": float(prob[1]),
                        "prob_neutral": float(prob[2]),
                    }
                )
        top1 = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        class_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0,
        ).tolist()
        class_precision = precision_score(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0,
        ).tolist()
        class_recall = recall_score(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0,
        ).tolist()
        confusion = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
        ).tolist()
        eval_loss = float(total_loss / max(total_num, 1.0))
        eval_tag = (
            "EVAL-TEST-FINAL"
            if eval_split == "test" and epoch is None
            else f"EVAL-{eval_split.upper()}"
        )
        confusion_text = (
            confusion
            if getattr(args, "log_confusion_matrix", True)
            else "disabled"
        )
        self.print_fn(
            "[{}] epoch={} loss={:.6f} acc={:.6f} macro_f1={:.6f} "
            "weighted_f1={:.6f} class_precision={} class_recall={} "
            "class_f1={} class_order=[positive,negative,neutral] "
            "confusion_matrix={}".format(
                eval_tag,
                "final" if epoch is None else epoch,
                eval_loss,
                top1,
                macro_f1,
                weighted_f1,
                [round(value, 6) for value in class_precision],
                [round(value, 6) for value in class_recall],
                [round(value, 6) for value in class_f1],
                confusion_text,
            )
        )
        
        self.ema.restore()
        self.model.train()
        return {
            'eval/loss': eval_loss,
            'eval/top-1-acc': top1,
            'eval/macro-f1': macro_f1,
            'eval/weighted-f1': weighted_f1,
            'eval/class-precision': class_precision,
            'eval/class-recall': class_recall,
            'eval/class-f1': class_f1,
            'eval/confusion-matrix': confusion,
            'eval/sample-records': sample_records,
        }

    @staticmethod
    def _label_name(label):
        return {
            0: "positive",
            1: "negative",
            2: "neutral",
        }.get(int(label), f"class_{int(label)}")

    def _save_evaluation_analysis(
        self,
        eval_dict,
        save_path,
        file_prefix,
        epoch,
        split_name,
    ):
        os.makedirs(save_path, exist_ok=True)
        records = list(eval_dict.get("eval/sample-records", []))
        class_names = [self._label_name(index) for index in range(self.num_classes)]

        total_by_true = {name: 0 for name in class_names}
        correct_by_true = {name: 0 for name in class_names}
        error_by_true = {name: 0 for name in class_names}
        predicted_counts = {name: 0 for name in class_names}
        error_transitions = {}
        for record in records:
            true_name = record["true_class"]
            pred_name = record["pred_class"]
            total_by_true[true_name] = total_by_true.get(true_name, 0) + 1
            predicted_counts[pred_name] = predicted_counts.get(pred_name, 0) + 1
            if record["correct"]:
                correct_by_true[true_name] = correct_by_true.get(true_name, 0) + 1
            else:
                error_by_true[true_name] = error_by_true.get(true_name, 0) + 1
                transition = f"{true_name}->{pred_name}"
                error_transitions[transition] = error_transitions.get(transition, 0) + 1

        correct_records = [record for record in records if record["correct"]]
        error_records = [record for record in records if not record["correct"]]
        class_f1 = eval_dict["eval/class-f1"]
        summary = {
            "split": split_name,
            "selection_metric": "accuracy",
            "best_epoch_index": None if epoch is None else int(epoch),
            "best_epoch": None if epoch is None else int(epoch) + 1,
            "accuracy": float(eval_dict["eval/top-1-acc"]),
            "macro_f1": float(eval_dict["eval/macro-f1"]),
            "positive_f1": float(class_f1[0]),
            "negative_f1": float(class_f1[1]),
            "neutral_f1": float(class_f1[2]),
            "weighted_f1": float(eval_dict["eval/weighted-f1"]),
            "loss": float(eval_dict["eval/loss"]),
            "total_samples": len(records),
            "correct_samples": len(correct_records),
            "error_samples": len(error_records),
            "total_by_true_class": total_by_true,
            "correct_by_true_class": correct_by_true,
            "error_by_true_class": error_by_true,
            "predicted_class_counts": predicted_counts,
            "error_transitions": dict(
                sorted(error_transitions.items(), key=lambda item: (-item[1], item[0]))
            ),
            "class_precision": {
                name: float(value)
                for name, value in zip(class_names, eval_dict["eval/class-precision"])
            },
            "class_recall": {
                name: float(value)
                for name, value in zip(class_names, eval_dict["eval/class-recall"])
            },
            "class_f1": {
                name: float(value)
                for name, value in zip(class_names, class_f1)
            },
            "confusion_matrix": eval_dict["eval/confusion-matrix"],
            "class_order": class_names,
        }

        summary_path = os.path.join(save_path, f"{file_prefix}_analysis.json")
        samples_path = os.path.join(save_path, f"{file_prefix}_samples.csv")
        correct_path = os.path.join(save_path, f"{file_prefix}_correct.csv")
        errors_path = os.path.join(save_path, f"{file_prefix}_errors.csv")
        history_path = os.path.join(save_path, "best_epoch_history.jsonl")

        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        pd.DataFrame(records).to_csv(samples_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(correct_records).to_csv(
            correct_path, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(error_records).to_csv(
            errors_path, index=False, encoding="utf-8-sig"
        )
        if file_prefix == "best_epoch":
            with open(history_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

        self.print_fn(
            "[BEST-EPOCH] epoch={} Accuracy={:.6f} Macro-F1={:.6f} "
            "Positive-F1={:.6f} Negative-F1={:.6f} Neutral-F1={:.6f} "
            "correct={} errors={} analysis={}".format(
                "final" if epoch is None else int(epoch) + 1,
                summary["accuracy"],
                summary["macro_f1"],
                summary["positive_f1"],
                summary["negative_f1"],
                summary["neutral_f1"],
                summary["correct_samples"],
                summary["error_samples"],
                summary_path,
            )
        )

    def save_model(self, save_name, save_path):
        # if self.it < 1000000:
        #     return
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()
        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        ucrf_runtime_beta = float(
            getattr(model_core, "ucrf_runtime_beta", 0.0)
        )

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'best_epoch': self.best_epoch,
                    'sa_dd_history': self.sa_dd_history,
                    'sa_dd_class_ema': self.sa_dd_class_ema,
                    'ucrf_runtime_beta': ucrf_runtime_beta,
                    'ema_model': ema_model},
                   save_filename)
        if self.num_classes == 10:
            tb_path = os.path.join(save_path, 'tensorboard')
            if not os.path.exists(tb_path):
                os.makedirs(tb_path, exist_ok=True)
            with open(os.path.join(save_path, 'tensorboard', 'lst_fix.pkl'), 'wb') as f:
                pickle.dump(self.lst, f)
            with open(os.path.join(save_path, 'tensorboard', 'abs_lst.pkl'), 'wb') as h:
                pickle.dump(self.abs_lst, h)
            with open(os.path.join(save_path, 'tensorboard', 'clsacc.pkl'), 'wb') as g:
                pickle.dump(self.clsacc, g)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        if hasattr(model_core, "set_ucrf_residual_beta"):
            model_core.set_ucrf_residual_beta(
                checkpoint.get(
                    "ucrf_runtime_beta",
                    getattr(model_core, "ucrf_runtime_beta", 0.0),
                )
            )
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.best_epoch = checkpoint.get('best_epoch')
        self.sa_dd_history = checkpoint.get('sa_dd_history', {})
        self.sa_dd_class_ema = checkpoint.get(
            'sa_dd_class_ema',
            self.sa_dd_class_ema,
        ).float().cpu()
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
