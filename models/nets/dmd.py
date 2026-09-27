"""
here is the mian backbone for DMD containing feature decoupling and multimodal transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torchvision.models as models
from math import sqrt

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Modal_Select(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Modal_Select, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class DCTRCalibrator(nn.Module):
    """
    Dual-calibrated tri-branch reliability estimator.

    The calibrator keeps Beta-smoothed correctness statistics for both the
    external evidence and the SCRD predictions. Statistics are independent for
    the multimodal, text, and image branches and are conditioned on predicted
    class and the tri-view evidence relation. Only labeled training samples
    update these buffers.
    """

    NUM_BRANCHES = 3
    NUM_RELATIONS = 5

    def __init__(self, num_classes, prior_mean=0.5, prior_strength=6.0):
        super(DCTRCalibrator, self).__init__()
        prior_mean = min(max(float(prior_mean), 1e-3), 1.0 - 1e-3)
        prior_strength = max(float(prior_strength), 1e-3)
        shape = (self.NUM_BRANCHES, int(num_classes), self.NUM_RELATIONS)
        alpha = torch.full(shape, prior_mean * prior_strength, dtype=torch.float)
        beta = torch.full(
            shape,
            (1.0 - prior_mean) * prior_strength,
            dtype=torch.float,
        )
        self.register_buffer("external_alpha", alpha.clone())
        self.register_buffer("external_beta", beta.clone())
        self.register_buffer("internal_alpha", alpha.clone())
        self.register_buffer("internal_beta", beta.clone())

    @torch.no_grad()
    def update(
        self,
        model_labels,
        evidence_labels,
        relation_group,
        y_true,
        evidence_valid,
        external_update_mask=None,
        internal_update_mask=None,
    ):
        relation_group = relation_group.long().clamp(
            min=0,
            max=self.NUM_RELATIONS - 1,
        )
        y_true = y_true.long()
        if external_update_mask is None:
            external_update_mask = torch.ones_like(
                y_true,
                dtype=torch.bool,
            )
        if internal_update_mask is None:
            internal_update_mask = torch.ones_like(
                y_true,
                dtype=torch.bool,
            )
        for branch in range(self.NUM_BRANCHES):
            model_pred = model_labels[:, branch].long()
            evidence_pred = evidence_labels[:, branch].long()

            self._index_update(
                self.internal_alpha[branch],
                self.internal_beta[branch],
                model_pred,
                relation_group,
                model_pred.eq(y_true),
                internal_update_mask.bool(),
            )
            self._index_update(
                self.external_alpha[branch],
                self.external_beta[branch],
                evidence_pred,
                relation_group,
                evidence_pred.eq(y_true),
                (
                    evidence_valid[:, branch].bool()
                    & external_update_mask.bool()
                ),
            )

    @staticmethod
    @torch.no_grad()
    def _index_update(alpha, beta, labels, relations, correct, valid):
        if not valid.any():
            return
        labels = labels[valid].clamp(min=0, max=alpha.shape[0] - 1)
        relations = relations[valid]
        correct = correct[valid].float()
        flat_index = labels * alpha.shape[1] + relations
        alpha.view(-1).index_add_(0, flat_index, correct)
        beta.view(-1).index_add_(0, flat_index, 1.0 - correct)

    def posterior(self, labels, relation_group, source):
        if source == "external":
            alpha, beta = self.external_alpha, self.external_beta
        elif source == "internal":
            alpha, beta = self.internal_alpha, self.internal_beta
        else:
            raise ValueError(f"Unknown DCTR calibration source: {source}")

        labels = labels.long().clamp(min=0, max=alpha.shape[1] - 1)
        relation_group = relation_group.long().clamp(
            min=0,
            max=self.NUM_RELATIONS - 1,
        )
        branch_index = torch.arange(
            self.NUM_BRANCHES,
            device=labels.device,
        ).view(1, -1).expand_as(labels)
        selected_alpha = alpha[
            branch_index,
            labels,
            relation_group.view(-1, 1).expand_as(labels),
        ]
        selected_beta = beta[
            branch_index,
            labels,
            relation_group.view(-1, 1).expand_as(labels),
        ]
        return selected_alpha / (selected_alpha + selected_beta).clamp_min(1e-6)

    def calibration_count(self):
        external_count = (
            self.external_alpha + self.external_beta
        ).sum().detach()
        internal_count = (
            self.internal_alpha + self.internal_beta
        ).sum().detach()
        return external_count, internal_count


class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        
        # 输入特征的维度
        self.in_dim = in_dim
        
        # 定义查询、键和值权重矩阵
        self.query_weight = nn.Linear(in_dim, in_dim)
        self.key_weight = nn.Linear(in_dim, in_dim)
        self.value_weight = nn.Linear(in_dim, in_dim)
        
        # 定义一个可学习的缩放参数
        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        # self.scale = sqrt(in_dim)
        
    def forward(self, x):
        # 计算查询、键和值
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        attn_weights = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, value)
        
        return output

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.visual_model = models.resnet18(pretrained=True)
        self.visual_model = nn.Sequential(*list(self.visual_model.children())[:-1])
        self.v_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)
        self.t_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)
        self.m_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)
        self.attention = Attention(in_dim=512)
        # self.modal_select_layer = nn.Linear(in_features=512*3, out_features=3)
        self.modal_select_layer = Modal_Select(input_size=512*3,hidden_size=512, num_classes=3)
        self.use_soft_modal_selector = bool(getattr(args, "use_soft_modal_selector", False))
        self.modal_selector_temperature = float(getattr(args, "modal_selector_temperature", 1.0))
        self.use_ucrf = bool(getattr(args, "use_ucrf", False))
        self.ucrf_residual_beta_max = float(
            getattr(args, "ucrf_residual_beta", 0.3)
        )
        # Training updates this value after the labeled-only selector warmup.
        # A fresh evaluation process uses the configured maximum by default.
        self.ucrf_runtime_beta = self.ucrf_residual_beta_max
        self.use_dctr_plf = (
            bool(getattr(args, "use_mllm_verification", False))
            and getattr(args, "mllm_action", "veto") == "dctr_plf"
        )
        self.dctr_calibrator = None
        if self.use_dctr_plf:
            self.dctr_calibrator = DCTRCalibrator(
                num_classes=args.num_classes,
                prior_mean=getattr(args, "dctr_prior_mean", 0.5),
                prior_strength=getattr(args, "dctr_prior_strength", 6.0),
            )
        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(768, 512, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 2.1 Modality-specific encoder
        self.encoder_s_l = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 2.2 Modality-invariant encoder
        self.encoder_c = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(512*2, 512, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(512*2, 512, kernel_size=1, padding=0, bias=False)

    def set_ucrf_residual_beta(self, beta):
        self.ucrf_runtime_beta = min(
            max(float(beta), 0.0),
            max(self.ucrf_residual_beta_max, 0.0),
        )

    def forward(self, image, text):
        # if self.use_bert:
        text = self.text_model(**text).pooler_output
        image = torch.flatten(self.visual_model(image), start_dim=1)
        # x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        # x_v = image.transpose(1, 2)

        text = self.proj_l(text.unsqueeze(2))
        image = self.proj_v(image.unsqueeze(2))
        # image = image.unsqueeze(2)
        # proj_x_v = self.proj_v(x_v)

        # Modality-specific feature
        s_l = self.encoder_s_l(text).squeeze(-1)
        s_v = self.encoder_s_v(image).squeeze(-1)
        pre_t = self.t_classifier(s_l)
        pre_v = self.v_classifier(s_v)

        # Modality-common feature
        c_l = self.encoder_c(text).squeeze(-1)
        c_v = self.encoder_c(image).squeeze(-1)
        c_m = c_l+c_v
        c_list = [c_l, c_v]
        pre_m = self.m_classifier(c_m)
        pre_m_in_v = self.v_classifier(c_m)
        pre_m_in_t = self.t_classifier(c_m)
        pre_v_in_m = self.m_classifier(s_v)
        pre_t_in_m = self.m_classifier(s_l)

        c_l_sim = c_l
        c_v_sim = c_v

        # decoder 
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1).unsqueeze(2))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1).unsqueeze(2))

        s_l_r = self.encoder_s_l(recon_l).squeeze(-1)
        s_v_r = self.encoder_s_v(recon_v).squeeze(-1)
        # Reuse the unimodal classifiers so EAD-A can impose semantic
        # supervision on the existing common-private reconstruction path.
        logits_text_recon = self.t_classifier(recon_l.squeeze(2))
        logits_image_recon = self.v_classifier(recon_v.squeeze(2))

        select_modal = self.modal_select_layer(torch.cat((s_l,s_v,c_m),dim=1))
        selector_temperature = max(self.modal_selector_temperature, 1e-6)
        modal_support = torch.softmax(select_modal / selector_temperature, dim=1)
        modal_index = torch.argmax(modal_support, dim=1)

        # attention
        s_l_att = s_l.unsqueeze(1)
        s_v_att = s_v.unsqueeze(1)
        c_m_att = c_m.unsqueeze(1)
        att_tensor = torch.cat((s_l_att, s_v_att, c_m_att), dim=1)
        att_m = self.attention(att_tensor)
        candidate_logits = self.m_classifier(
            att_m.reshape(-1, att_m.size(-1))
        ).reshape(att_m.size(0), att_m.size(1), -1)
        batch_index = torch.arange(att_m.size(0), device=att_m.device)
        hard_select_m = att_m[batch_index, modal_index]
        selector_entropy = -(
            modal_support.clamp_min(1e-8)
            * modal_support.clamp_min(1e-8).log()
        ).sum(dim=1)
        selector_confidence = (
            1.0 - selector_entropy / torch.log(
                modal_support.new_tensor(float(modal_support.size(1)))
            )
        ).clamp(min=0.0, max=1.0)
        if self.use_ucrf:
            soft_select_m = (
                modal_support.unsqueeze(-1) * att_m
            ).sum(dim=1)
            # Uncertain gates fall back toward the common candidate; confident
            # gates retain the learned soft mixture. The residual form keeps
            # beta=0 exactly equivalent to the original hard SCRD selector.
            common_fallback = att_m[:, 2]
            residual_target = (
                (1.0 - selector_confidence).unsqueeze(-1)
                * common_fallback
                + selector_confidence.unsqueeze(-1) * soft_select_m
            )
            select_m = hard_select_m + float(
                self.ucrf_runtime_beta
            ) * (
                residual_target - hard_select_m
            )
        elif self.use_soft_modal_selector:
            select_m = (modal_support.unsqueeze(-1) * att_m).sum(dim=1)
        else:
            select_m = hard_select_m
        # select_m = select_m.unsqueeze(1)
        # modal_index = modal_index.unsqueeze(0)
        # att_m = torch.gather(att_m,1,modal_index)
        # [:,modal_index,:]
        pre_m_att = self.m_classifier(select_m)

        res = {
            # 'origin_l': proj_x_l,
            # 'origin_v': proj_x_v,
            'origin_l': text,
            'origin_v': image,
            's_l': s_l,
            's_v': s_v,

            'c_l': c_l,
            'c_v': c_v,

            's_l_r': s_l_r,
            's_v_r': s_v_r,

            'recon_l': recon_l,
            'recon_v': recon_v,

            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,

            'att_m': att_m,
            'modal_select_logits': select_modal,
            'modal_support': modal_support,
            'soft_modal_selector': self.use_soft_modal_selector,
            'candidate_logits': candidate_logits,
            'ucrf_selector_confidence': selector_confidence,
            'ucrf_residual_beta': modal_support.new_full(
                (modal_support.size(0),),
                float(self.ucrf_runtime_beta) if self.use_ucrf else 0.0,
            ),
            # 'modal_index': modal_index,

            'pre_t': pre_t,
            'pre_v': pre_v,
            'pre_m': pre_m,
            'pre_m_att': pre_m_att,

            'logits_common': pre_m,
            'logits_text_private': pre_t,
            'logits_image_private': pre_v,
            'logits_fusion': pre_m_att,
            'logits_text_recon': logits_text_recon,
            'logits_image_recon': logits_image_recon,
            'z_common': c_m,
            'z_text_private': s_l,
            'z_image_private': s_v,

            'pre_m_in_t': pre_m_in_t,
            'pre_m_in_v': pre_m_in_v,
            'pre_v_in_m': pre_v_in_m,
            'pre_t_in_m': pre_t_in_m,

            'modal_index': modal_index
        }
        return res
