# -*- coding: utf-8 -*-

import json
import random
from collections import Counter

import torch


def parse_sentiment_label_map(mapping_text):
    mapping = {}
    for item in mapping_text.split(","):
        if not item.strip():
            continue
        name, value = item.split(":")
        mapping[normalize_sentiment(name)] = int(value)
    required = {"positive", "negative", "neutral"}
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"mllm label map misses keys: {sorted(missing)}")
    return mapping


def normalize_sentiment(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if text.startswith("pos"):
        return "positive"
    if text.startswith("neg"):
        return "negative"
    if text.startswith("neu"):
        return "neutral"
    return text


def sample_ids_to_list(sample_ids):
    if torch.is_tensor(sample_ids):
        return [str(x) for x in sample_ids.detach().cpu().view(-1).tolist()]
    if isinstance(sample_ids, (list, tuple)):
        out = []
        for item in sample_ids:
            if torch.is_tensor(item):
                item = item.item()
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            out.append(str(item))
        return out
    return [str(sample_ids)]


class MLLMEvidenceVerifier:
    """
    External tri-view MLLM evidence for pseudo-label reliability verification.

    By default the verifier never replaces SCRD pseudo-labels. It returns an
    additional mask/weight that decides whether SCRD pseudo-label candidates are
    reliable enough to enter unsupervised training. The override action is kept
    only for controlled ablations.
    """

    def __init__(
        self,
        path="",
        enabled=False,
        mode="mm",
        label_map_text="positive:0,negative:1,neutral:2",
        missing_policy="reject",
        shuffle=False,
        seed=1,
    ):
        self.path = path or ""
        self.enabled = bool(enabled)
        self.mode = mode
        self.label_map = parse_sentiment_label_map(label_map_text)
        self.num_classes = max(self.label_map.values()) + 1
        self.neutral_id = self.label_map["neutral"]
        self.positive_id = self.label_map["positive"]
        self.negative_id = self.label_map["negative"]
        self.missing_policy = missing_policy
        self.evidence = {}

        if not self.enabled:
            return
        if not self.path:
            raise ValueError("--use_mllm_verification requires --mllm_evidence_path")

        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(row.get("id", "")).strip()
                if not sid:
                    raise ValueError(f"missing id in evidence line {line_no}")
                item = {
                    "image": self._sent_to_id(row.get("image_sentiment")),
                    "text": self._sent_to_id(row.get("text_sentiment")),
                    "multimodal": self._sent_to_id(row.get("multimodal_sentiment")),
                    "image_dist": self._dist_to_vec(row.get("image_distribution")),
                    "text_dist": self._dist_to_vec(row.get("text_distribution")),
                    "multimodal_dist": self._dist_to_vec(row.get("multimodal_distribution")),
                }
                rows.append((sid, item))

        if shuffle:
            rng = random.Random(seed)
            ids = [sid for sid, _ in rows]
            values = [item for _, item in rows]
            rng.shuffle(values)
            rows = list(zip(ids, values))

        self.evidence = dict(rows)

    def _sent_to_id(self, value):
        sentiment = normalize_sentiment(value)
        if sentiment not in self.label_map:
            return -1
        return self.label_map[sentiment]

    def _dist_to_vec(self, value):
        if not isinstance(value, dict):
            return None
        vec = [0.0 for _ in range(self.num_classes)]
        total = 0.0
        for name, idx in self.label_map.items():
            try:
                prob = float(value.get(name, 0.0))
            except (TypeError, ValueError):
                prob = 0.0
            prob = max(prob, 0.0)
            vec[idx] = prob
            total += prob
        if total <= 0.0:
            return None
        return [prob / total for prob in vec]

    def lookup(self, sample_ids, device):
        ids = sample_ids_to_list(sample_ids)
        image, text, multimodal = [], [], []
        image_dist, text_dist, multimodal_dist = [], [], []
        has_image_dist, has_text_dist, has_multimodal_dist = [], [], []
        for sid in ids:
            item = self.evidence.get(str(sid))
            if item is None:
                image.append(-1)
                text.append(-1)
                multimodal.append(-1)
                image_dist.append([0.0 for _ in range(self.num_classes)])
                text_dist.append([0.0 for _ in range(self.num_classes)])
                multimodal_dist.append([0.0 for _ in range(self.num_classes)])
                has_image_dist.append(False)
                has_text_dist.append(False)
                has_multimodal_dist.append(False)
            else:
                image.append(item["image"])
                text.append(item["text"])
                multimodal.append(item["multimodal"])
                for key, out, flags in (
                    ("image_dist", image_dist, has_image_dist),
                    ("text_dist", text_dist, has_text_dist),
                    ("multimodal_dist", multimodal_dist, has_multimodal_dist),
                ):
                    dist = item.get(key)
                    if dist is None:
                        out.append([0.0 for _ in range(self.num_classes)])
                        flags.append(False)
                    else:
                        out.append(dist)
                        flags.append(True)
        return {
            "image": torch.tensor(image, dtype=torch.long, device=device),
            "text": torch.tensor(text, dtype=torch.long, device=device),
            "multimodal": torch.tensor(multimodal, dtype=torch.long, device=device),
            "image_dist": torch.tensor(image_dist, dtype=torch.float, device=device),
            "text_dist": torch.tensor(text_dist, dtype=torch.float, device=device),
            "multimodal_dist": torch.tensor(multimodal_dist, dtype=torch.float, device=device),
            "has_image_dist": torch.tensor(has_image_dist, dtype=torch.bool, device=device),
            "has_text_dist": torch.tensor(has_text_dist, dtype=torch.bool, device=device),
            "has_multimodal_dist": torch.tensor(has_multimodal_dist, dtype=torch.bool, device=device),
        }

    def health_summary(self):
        if not self.enabled:
            return "MLLM verification disabled."
        inv = {v: k for k, v in self.label_map.items()}
        counters = {"image": Counter(), "text": Counter(), "multimodal": Counter()}
        relation = Counter()
        for item in self.evidence.values():
            for key in counters:
                counters[key][inv.get(item[key], "unknown")] += 1
            img = item["image"]
            txt = item["text"]
            mm = item["multimodal"]
            if img == txt:
                relation["image_eq_text"] += 1
            if img == self.neutral_id and txt != self.neutral_id:
                relation["image_neutral_text_emotional"] += 1
            if txt == self.neutral_id and img != self.neutral_id:
                relation["text_neutral_image_emotional"] += 1
            if img != self.neutral_id and txt != self.neutral_id and img != txt:
                relation["image_text_opposite"] += 1
            if mm == img:
                relation["multimodal_eq_image"] += 1
            if mm == txt:
                relation["multimodal_eq_text"] += 1
        return (
            f"MLLM evidence loaded: n={len(self.evidence)}, mode={self.mode}, "
            f"image_dist={dict(counters['image'])}, text_dist={dict(counters['text'])}, "
            f"multimodal_dist={dict(counters['multimodal'])}, relation={dict(relation)}"
        )

    def ec_pfd_targets(self, sample_ids, y_m, fallback_prob=None, device=None):
        """
        Build detached EC-PFD targets from offline evidence.

        The target remains anchored to SCRD pseudo-labels: external multimodal
        distributions are used only when their top label agrees with the SCRD
        pseudo-label. Otherwise the caller-provided SCRD weak distribution is
        used as the soft pseudo-label target.
        """
        if device is None:
            device = y_m.device
        if fallback_prob is None:
            target_prob = torch.zeros(
                y_m.shape[0],
                self.num_classes,
                dtype=torch.float,
                device=device,
            )
            target_prob.scatter_(1, y_m.clamp(min=0, max=self.num_classes - 1).view(-1, 1), 1.0)
        else:
            target_prob = fallback_prob.float().to(device).clone()

        neutral_uncertain = torch.zeros_like(y_m, dtype=torch.bool, device=device)
        external_target_ratio = 0.0
        if not self.enabled or self.mode == "off":
            return target_prob, neutral_uncertain, {
                "ec_pfd_external_target_ratio": external_target_ratio,
                "ec_pfd_neutral_uncertain_ratio": 0.0,
            }

        ev = self.lookup(sample_ids, device=device)
        ev_t = ev["text"]
        ev_v = ev["image"]
        ev_m = ev["multimodal"]
        ev_m_dist = ev["multimodal_dist"]
        has_m_dist = ev["has_multimodal_dist"]

        valid_t = ev_t.ge(0)
        valid_v = ev_v.ge(0)
        valid_m = ev_m.ge(0)
        valid_all = valid_t & valid_v & valid_m
        neutral_consensus = (
            valid_all
            & ev_t.eq(self.neutral_id)
            & ev_v.eq(self.neutral_id)
            & ev_m.eq(self.neutral_id)
        )
        neutral_uncertain = valid_m & ev_m.eq(self.neutral_id)
        neutral_uncertain = neutral_uncertain | neutral_consensus

        external_ready = has_m_dist & valid_m & ev_m.eq(y_m)
        if external_ready.any():
            target_prob = torch.where(external_ready.view(-1, 1), ev_m_dist, target_prob)

        total = max(int(y_m.numel()), 1)
        log = {
            "ec_pfd_external_target_ratio": float(external_ready.float().sum().detach().cpu()) / float(total),
            "ec_pfd_neutral_uncertain_ratio": float(neutral_uncertain.float().sum().detach().cpu()) / float(total),
        }
        return target_prob, neutral_uncertain, log

    def msd_evidence(self, sample_ids, device):
        """
        Return detached tri-view distributions for modality support distillation.

        The modality order used by MSD is text, image, multimodal, matching the
        selector candidates (s_l, s_v, c_m) in DMD.
        """
        ids = sample_ids_to_list(sample_ids)
        batch_size = len(ids)
        zeros = torch.zeros(
            batch_size,
            self.num_classes,
            dtype=torch.float,
            device=device,
        )
        false_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if not self.enabled or self.mode == "off":
            return {
                "text_dist": zeros.detach(),
                "image_dist": zeros.detach(),
                "multimodal_dist": zeros.detach(),
                "valid_text": false_mask.detach(),
                "valid_image": false_mask.detach(),
                "valid_multimodal": false_mask.detach(),
                "valid_all": false_mask.detach(),
                "neutral_consensus": false_mask.detach(),
                "opposite_conflict": false_mask.detach(),
            }

        ev = self.lookup(sample_ids, device=device)
        valid_text = ev["has_text_dist"] & ev["text"].ge(0)
        valid_image = ev["has_image_dist"] & ev["image"].ge(0)
        valid_multimodal = ev["has_multimodal_dist"] & ev["multimodal"].ge(0)
        valid_all = valid_text & valid_image & valid_multimodal
        valid_labels = ev["text"].ge(0) & ev["image"].ge(0) & ev["multimodal"].ge(0)
        neutral_consensus = (
            valid_labels
            & ev["text"].eq(self.neutral_id)
            & ev["image"].eq(self.neutral_id)
            & ev["multimodal"].eq(self.neutral_id)
        )
        opposite_conflict = ev["text"].ge(0) & ev["image"].ge(0) & (
            (ev["text"].eq(self.positive_id) & ev["image"].eq(self.negative_id))
            | (ev["text"].eq(self.negative_id) & ev["image"].eq(self.positive_id))
        )
        return {
            "text_dist": ev["text_dist"].detach(),
            "image_dist": ev["image_dist"].detach(),
            "multimodal_dist": ev["multimodal_dist"].detach(),
            "valid_text": valid_text.detach(),
            "valid_image": valid_image.detach(),
            "valid_multimodal": valid_multimodal.detach(),
            "valid_all": valid_all.detach(),
            "neutral_consensus": neutral_consensus.detach(),
            "opposite_conflict": opposite_conflict.detach(),
        }

    def dctr_evidence(self, sample_ids, device):
        """
        Return detached evidence for DCTR-PLF.

        Branch order is multimodal, text, image. Relation groups are:
        0 other, 1 emotional tri-consensus, 2 neutral tri-consensus,
        3 positive-negative opposite conflict, 4 one-neutral imbalance.
        """
        ev = self.lookup(sample_ids, device=device)
        labels = torch.stack(
            [ev["multimodal"], ev["text"], ev["image"]],
            dim=1,
        )
        distributions = torch.stack(
            [
                ev["multimodal_dist"],
                ev["text_dist"],
                ev["image_dist"],
            ],
            dim=1,
        )
        valid = torch.stack(
            [
                ev["has_multimodal_dist"] & ev["multimodal"].ge(0),
                ev["has_text_dist"] & ev["text"].ge(0),
                ev["has_image_dist"] & ev["image"].ge(0),
            ],
            dim=1,
        )

        valid_labels = (
            ev["text"].ge(0)
            & ev["image"].ge(0)
            & ev["multimodal"].ge(0)
        )
        tri_consensus = (
            valid_labels
            & ev["text"].eq(ev["image"])
            & ev["image"].eq(ev["multimodal"])
        )
        neutral_consensus = tri_consensus & ev["multimodal"].eq(
            self.neutral_id
        )
        emotional_consensus = tri_consensus & (~neutral_consensus)
        opposite_conflict = (
            ev["text"].ge(0)
            & ev["image"].ge(0)
            & (
                (
                    ev["text"].eq(self.positive_id)
                    & ev["image"].eq(self.negative_id)
                )
                | (
                    ev["text"].eq(self.negative_id)
                    & ev["image"].eq(self.positive_id)
                )
            )
        )
        one_neutral = (
            ev["text"].ge(0)
            & ev["image"].ge(0)
            & (
                (
                    ev["text"].eq(self.neutral_id)
                    & ev["image"].ne(self.neutral_id)
                )
                | (
                    ev["image"].eq(self.neutral_id)
                    & ev["text"].ne(self.neutral_id)
                )
            )
        )
        relation_group = torch.zeros(
            labels.shape[0],
            dtype=torch.long,
            device=device,
        )
        relation_group = relation_group.masked_fill(emotional_consensus, 1)
        relation_group = relation_group.masked_fill(neutral_consensus, 2)
        relation_group = relation_group.masked_fill(opposite_conflict, 3)
        relation_group = relation_group.masked_fill(one_neutral, 4)

        return {
            "labels": labels.detach(),
            "distributions": distributions.detach(),
            "valid": valid.detach(),
            "relation_group": relation_group.detach(),
            "emotional_consensus": emotional_consensus.detach(),
            "neutral_consensus": neutral_consensus.detach(),
            "opposite_conflict": opposite_conflict.detach(),
            "one_neutral": one_neutral.detach(),
        }

    def diagnostic_log(
        self,
        base_mask,
        final_mask,
        labels,
        y_true=None,
        loss_weight=None,
    ):
        return self._log(
            base_mask=base_mask,
            final_mask=final_mask,
            y_m=labels,
            y_true=y_true,
            loss_weight=loss_weight,
        )

    def verify(
        self,
        base_mask,
        sample_ids,
        y_t,
        y_v,
        y_m,
        y_true=None,
        verify_mask=None,
        action="veto",
        soft_weight=0.5,
        ecs_conf_threshold=0.75,
        hybrid_config=None,
    ):
        base_mask = base_mask.bool()
        if verify_mask is None:
            verify_mask = torch.ones_like(base_mask, dtype=torch.bool)
        else:
            verify_mask = verify_mask.bool()

        loss_weight = torch.ones_like(y_m, dtype=torch.float)
        override_labels = y_m.clone()

        if not self.enabled or self.mode == "off":
            return base_mask, self._log(base_mask, base_mask, y_m, y_true, loss_weight), loss_weight, override_labels

        ev = self.lookup(sample_ids, device=y_m.device)
        ev_t = ev["text"]
        ev_v = ev["image"]
        ev_m = ev["multimodal"]
        ev_t_dist = ev["text_dist"]
        ev_v_dist = ev["image_dist"]
        ev_m_dist = ev["multimodal_dist"]
        has_t_dist = ev["has_text_dist"]
        has_v_dist = ev["has_image_dist"]
        has_m_dist = ev["has_multimodal_dist"]
        ev_m_conf = ev_m_dist.max(dim=1).values
        top2_m = torch.topk(ev_m_dist, k=min(2, ev_m_dist.shape[1]), dim=1).values
        if top2_m.shape[1] > 1:
            ev_m_margin = top2_m[:, 0] - top2_m[:, 1]
        else:
            ev_m_margin = torch.zeros_like(ev_m_conf)

        valid_t = ev_t.ge(0)
        valid_v = ev_v.ge(0)
        valid_m = ev_m.ge(0)
        valid_all = valid_t & valid_v & valid_m

        mm_agree = valid_m & y_m.eq(ev_m)
        internal_conflict = y_t.ne(y_v)
        external_conflict = valid_t & valid_v & ev_t.ne(self.neutral_id) & ev_v.ne(self.neutral_id) & ev_t.ne(ev_v)
        opposite_conflict = valid_t & valid_v & (
            (ev_t.eq(self.positive_id) & ev_v.eq(self.negative_id))
            | (ev_t.eq(self.negative_id) & ev_v.eq(self.positive_id))
        )
        emotional_consensus = valid_all & ev_t.eq(ev_v) & ev_v.eq(ev_m) & ev_m.ne(self.neutral_id)
        neutral_consensus = valid_all & ev_t.eq(self.neutral_id) & ev_v.eq(self.neutral_id) & ev_m.eq(self.neutral_id)
        modality_imbalance = valid_t & valid_v & (
            (ev_t.eq(self.neutral_id) & ev_v.ne(self.neutral_id))
            | (ev_v.eq(self.neutral_id) & ev_t.ne(self.neutral_id))
        )
        text_dominant = valid_t & valid_v & ev_v.eq(self.neutral_id) & ev_t.ne(self.neutral_id) & y_m.eq(ev_t)
        image_dominant = valid_t & valid_v & ev_t.eq(self.neutral_id) & ev_v.ne(self.neutral_id) & y_m.eq(ev_v)

        if self.mode == "mm":
            evidence_ok = mm_agree
            required_valid = valid_m
        elif self.mode == "tri":
            evidence_ok = valid_all & (
                mm_agree
                | ((~mm_agree) & (~(internal_conflict & external_conflict)) & (text_dominant | image_dominant))
            )
            required_valid = valid_all
        elif self.mode == "strict":
            evidence_ok = valid_all & y_t.eq(ev_t) & y_v.eq(ev_v) & y_m.eq(ev_m)
            required_valid = valid_all
        elif self.mode == "img_text":
            evidence_ok = valid_t & valid_v & (~(internal_conflict & external_conflict))
            required_valid = valid_t & valid_v
        else:
            raise ValueError(f"Unknown mllm verification mode: {self.mode}")

        if self.missing_policy == "pass":
            evidence_ok = evidence_ok | (~required_valid)
        elif self.missing_policy != "reject":
            raise ValueError(f"Unknown mllm missing policy: {self.missing_policy}")

        verified = verify_mask
        reject_by_evidence = verified & (~evidence_ok)
        hybrid_stats = {
            "hybrid_support_mean": 0.0,
            "hybrid_q_conf_mean": 0.0,
            "hybrid_q_margin_mean": 0.0,
            "hybrid_strong_support_ratio": 0.0,
            "hybrid_uncertain_ratio": 0.0,
            "hybrid_high_conflict_ratio": 0.0,
            "hybrid_opposite_conflict_ratio": 0.0,
            "hybrid_neutral_limited_ratio": 0.0,
        }
        if action == "veto":
            final_mask = base_mask & ((~verified) | evidence_ok)
        elif action == "soft_weight":
            final_mask = base_mask
            loss_weight = loss_weight.masked_fill(base_mask & reject_by_evidence, float(soft_weight))
        elif action == "ecs_v1":
            pass_agree = base_mask & mm_agree
            pass_disagree = base_mask & valid_m & (~mm_agree)
            pass_missing = base_mask & (~valid_m)
            rescue = (
                (~base_mask)
                & emotional_consensus
                & y_m.eq(ev_m)
                & has_m_dist
                & ev_m_conf.ge(float(ecs_conf_threshold))
            )
            final_mask = base_mask | rescue
            loss_weight = torch.zeros_like(y_m, dtype=torch.float)
            loss_weight = loss_weight.masked_fill(pass_agree | pass_missing, 1.0)
            loss_weight = loss_weight.masked_fill(pass_disagree | rescue, float(soft_weight))
            reject_by_evidence = (~base_mask) & opposite_conflict
        elif action == "hybrid_cerw":
            cfg = {
                "support_high": 0.65,
                "support_low": 0.30,
                "qwen_conf_high": 0.75,
                "weight_agree": 1.0,
                "weight_uncertain": 0.5,
                "weight_high_conflict": 0.0,
                "weight_opposite_conflict": 0.0,
                "neutral_max_weight": 0.5,
                "use_distribution": True,
                "use_opposite_conflict": True,
            }
            if hybrid_config:
                cfg.update(hybrid_config)

            use_distribution = bool(cfg["use_distribution"])
            use_opposite_conflict = bool(cfg["use_opposite_conflict"])
            if use_distribution:
                required_valid = valid_t & valid_v & valid_m & has_t_dist & has_v_dist & has_m_dist
                y_index = y_m.clamp(min=0, max=self.num_classes - 1).view(-1, 1)
                support = ev_m_dist.gather(1, y_index).squeeze(1)
                q_conf = ev_m_conf
                q_margin = ev_m_margin
            else:
                required_valid = valid_m
                support = mm_agree.float()
                q_conf = valid_m.float()
                q_margin = torch.zeros_like(q_conf)

            final_mask = base_mask.clone()
            loss_weight = torch.ones_like(y_m, dtype=torch.float)
            target = base_mask & verified
            valid_target = target & required_valid
            if self.missing_policy == "reject":
                final_mask = final_mask & ((~target) | required_valid)
                loss_weight = loss_weight.masked_fill(target & (~required_valid), 0.0)
            elif self.missing_policy != "pass":
                raise ValueError(f"Unknown mllm missing policy: {self.missing_policy}")

            strong_support = valid_target & support.ge(float(cfg["support_high"]))
            high_conflict = (
                valid_target
                & (~strong_support)
                & support.lt(float(cfg["support_low"]))
                & q_conf.ge(float(cfg["qwen_conf_high"]))
            )
            if use_opposite_conflict:
                opposite_rule = valid_target & (~strong_support) & (~high_conflict) & opposite_conflict
            else:
                opposite_rule = torch.zeros_like(valid_target, dtype=torch.bool)
            uncertain = valid_target & (~strong_support) & (~high_conflict) & (~opposite_rule)

            loss_weight = loss_weight.masked_fill(strong_support, float(cfg["weight_agree"]))
            loss_weight = loss_weight.masked_fill(uncertain, float(cfg["weight_uncertain"]))
            loss_weight = loss_weight.masked_fill(high_conflict, float(cfg["weight_high_conflict"]))
            loss_weight = loss_weight.masked_fill(opposite_rule, float(cfg["weight_opposite_conflict"]))

            neutral_limited = valid_target & (ev_m.eq(self.neutral_id) | neutral_consensus)
            neutral_cap = float(cfg["neutral_max_weight"])
            if neutral_cap >= 0.0:
                capped = torch.full_like(loss_weight, neutral_cap)
                loss_weight = torch.where(neutral_limited, torch.minimum(loss_weight, capped), loss_weight)

            final_mask = final_mask & (~(target & loss_weight.le(0.0)))
            reject_by_evidence = verified & base_mask & (~final_mask)

            target_count = target.float().sum().clamp_min(1.0)
            hybrid_stats = {
                "hybrid_support_mean": self._masked_mean(support, valid_target),
                "hybrid_q_conf_mean": self._masked_mean(q_conf, valid_target),
                "hybrid_q_margin_mean": self._masked_mean(q_margin, valid_target),
                "hybrid_strong_support_ratio": float(strong_support.float().sum().detach().cpu() / target_count.detach().cpu()),
                "hybrid_uncertain_ratio": float(uncertain.float().sum().detach().cpu() / target_count.detach().cpu()),
                "hybrid_high_conflict_ratio": float(high_conflict.float().sum().detach().cpu() / target_count.detach().cpu()),
                "hybrid_opposite_conflict_ratio": float(opposite_rule.float().sum().detach().cpu() / target_count.detach().cpu()),
                "hybrid_neutral_limited_ratio": float(neutral_limited.float().sum().detach().cpu() / target_count.detach().cpu()),
            }
        elif action == "dctr_plf":
            # DCTR-PLF estimates three independent continuous reliabilities in
            # the training loop. The verifier only exposes evidence here and
            # leaves the original SCRD candidate mask unchanged.
            final_mask = base_mask
        elif action == "override":
            # This option is provided for ablation only. The default method keeps
            # SCRD pseudo-labels and uses MLLM evidence only as a verifier.
            final_mask = base_mask
            override_ready = verified & required_valid
            override_labels = torch.where(override_ready, ev_m, override_labels)
        else:
            raise ValueError(f"Unknown mllm action: {action}")

        final_labels_for_log = override_labels if action == "override" else y_m
        log = self._log(base_mask, final_mask, final_labels_for_log, y_true, loss_weight)
        verified_count = verified.float().sum().clamp_min(1.0)
        verified_base = verified & base_mask
        verified_base_count = verified_base.float().sum().clamp_min(1.0)
        log.update({
            "missing_ratio": float((~required_valid).float().mean().detach().cpu()),
            "mm_agree_ratio": float(mm_agree.float().mean().detach().cpu()),
            "mllm_used_ratio": float(verified.float().mean().detach().cpu()),
            "mllm_agree_ratio_on_verified": float((mm_agree & verified).float().sum().detach().cpu() / verified_count.detach().cpu()),
            "mllm_reject_ratio_on_verified": float((reject_by_evidence & verified_base).float().sum().detach().cpu() / verified_base_count.detach().cpu()),
            "internal_conflict_ratio": float(internal_conflict.float().mean().detach().cpu()),
            "external_conflict_ratio": float(external_conflict.float().mean().detach().cpu()),
            "ecs_rescued_coverage": self._mask_ratio((~base_mask) & final_mask),
            "ecs_effective_coverage": self._mask_ratio(final_mask),
            "ecs_emotional_consensus_count": int(emotional_consensus.sum().detach().cpu()),
            "ecs_neutral_consensus_count": int(neutral_consensus.sum().detach().cpu()),
            "ecs_modality_imbalance_count": int(modality_imbalance.sum().detach().cpu()),
            "ecs_opposite_conflict_count": int(opposite_conflict.sum().detach().cpu()),
            "ecs_qwen_agree_pass_count": int((base_mask & mm_agree).sum().detach().cpu()),
            "ecs_qwen_disagree_pass_count": int((base_mask & valid_m & (~mm_agree)).sum().detach().cpu()),
            "ecs_rescued_pseudo_acc": self._masked_acc(final_labels_for_log, y_true, (~base_mask) & final_mask),
            "ecs_opposite_conflict_pseudo_acc": self._masked_acc(final_labels_for_log, y_true, opposite_conflict),
        })
        log.update(hybrid_stats)
        return final_mask, log, loss_weight, override_labels

    def _log(self, base_mask, final_mask, y_m, y_true=None, loss_weight=None):
        total = max(int(base_mask.numel()), 1)
        base_count = int(base_mask.sum().detach().cpu())
        final_count = int(final_mask.sum().detach().cpu())
        rejected = base_mask & (~final_mask)
        if loss_weight is None:
            loss_weight = torch.ones_like(y_m, dtype=torch.float)
        weighted_mask = final_mask.float() * loss_weight.float()

        log = {
            "base_coverage": base_count / float(total),
            "final_coverage": final_count / float(total),
            "weighted_coverage": float(weighted_mask.sum().detach().cpu()) / float(total),
            "retention": final_count / float(max(base_count, 1)),
            "reject_error_rate": 0.0,
            "base_pseudo_acc": 0.0,
            "final_pseudo_acc": 0.0,
            "weighted_pseudo_acc": 0.0,
            "effective_signal": 0.0,
            "final_pred_dist": self._masked_distribution(y_m, final_mask),
            "weighted_final_pred_dist": self._weighted_distribution(y_m, weighted_mask),
            "rejected_count": int(rejected.sum().detach().cpu()),
        }
        if y_true is not None:
            correct = y_m.eq(y_true)
            log["base_pseudo_acc"] = self._masked_mean(correct.float(), base_mask)
            log["final_pseudo_acc"] = self._masked_mean(correct.float(), final_mask)
            log["weighted_pseudo_acc"] = self._weighted_mean(correct.float(), weighted_mask)
            log["reject_error_rate"] = self._masked_mean((~correct).float(), rejected)
            log["effective_signal"] = log["weighted_coverage"] * log["weighted_pseudo_acc"]
        return log

    def _mask_ratio(self, mask):
        total = max(int(mask.numel()), 1)
        return float(mask.float().sum().detach().cpu()) / float(total)

    def _masked_acc(self, labels, y_true, mask):
        if y_true is None:
            return 0.0
        return self._masked_mean(labels.eq(y_true).float(), mask)

    def _masked_mean(self, values, mask):
        denom = mask.float().sum().clamp_min(1.0)
        return float((values * mask.float()).sum().detach().cpu() / denom.detach().cpu())

    def _weighted_mean(self, values, weights):
        denom = weights.float().sum().clamp_min(1.0)
        return float((values * weights.float()).sum().detach().cpu() / denom.detach().cpu())

    def _masked_distribution(self, labels, mask):
        counts = []
        for cls in range(len(self.label_map)):
            counts.append(int(((labels == cls) & mask).sum().detach().cpu()))
        total = max(sum(counts), 1)
        return [round(c / float(total), 6) for c in counts]

    def _weighted_distribution(self, labels, weights):
        counts = []
        for cls in range(len(self.label_map)):
            counts.append(float((weights * labels.eq(cls).float()).sum().detach().cpu()))
        total = max(sum(counts), 1.0)
        return [round(c / float(total), 6) for c in counts]
