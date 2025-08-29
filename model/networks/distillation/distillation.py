import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_weighted_kl_divergence(kl_divergence_row, weight_row):
    weighted_kl_divergence_row = kl_divergence_row * weight_row
    return torch.sum(weighted_kl_divergence_row)


def distillation_loss(logits_l, logits_h, weight):
    logits_l = F.log_softmax(logits_l, dim=1)
    logits_h = F.softmax(logits_h, dim=1)
    kl_divergence = F.kl_div(logits_l, logits_h, reduction='none')
    weight = weight.T
    summed_weighted_kl_divergence = torch.stack(
        [calculate_weighted_kl_divergence(kl_divergence[i], weight[i]) for i in range(logits_l.shape[0])])
    sum_c_f = torch.sum(weight)
    epsilon = 1e-8
    sum_c_f = sum_c_f + epsilon
    loss = torch.mean(summed_weighted_kl_divergence) / sum_c_f
    return loss


class Distillation(nn.Module):
    def __init__(self, args, hidden_dim):
        super(Distillation, self).__init__()
        self.args = args
        self.temp_proto = nn.Parameter(torch.tensor(10., requires_grad=True))
        self.method = 'dot'
        self.temp = nn.Parameter(torch.tensor(1., requires_grad=True))
        self.temp_dct = nn.Parameter(torch.tensor(1., requires_grad=True))
        self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def compute_logits(self, feat, proto, metric='dot', temp=1.0):
        assert feat.dim() == proto.dim()
        if feat.dim() == 2:
            if metric == 'dot':
                logits = torch.mm(feat, proto.t())
            elif metric == 'cos':
                logits = 1 - torch.mm(F.normalize(feat, dim=-1), F.normalize(proto, dim=-1).t())
            elif metric == 'sqr':
                logits = -(feat.unsqueeze(1) - proto.unsqueeze(0)).pow(2).sum(dim=-1)

        elif feat.dim() == 3:
            if metric == 'dot':
                logits = torch.bmm(feat, proto.permute(0, 2, 1))
            elif metric == 'cos':
                logits = 1 - torch.bmm(F.normalize(feat, dim=-1), F.normalize(proto, dim=-1).permute(0, 2, 1))
            elif metric == 'sqr':
                logits = -(feat.unsqueeze(2) - proto.unsqueeze(1)).pow(2).sum(dim=-1)

        norm = logits.norm(p=2, dim=1, keepdim=True)
        logits = logits / (norm + 1e-6)
        return logits * temp

    def proto_refine(self, logits, feat, proto):
        absolute_certainty, _ = torch.max(logits, dim=2)
        if absolute_certainty.shape[1] < self.args.k_value:
            k = absolute_certainty.shape[1]
        else:
            k = self.args.k_value
        max_values, max_indices = torch.topk(absolute_certainty, k=k, dim=1, largest=True)
        weighted_features = torch.zeros_like(proto[0, :], dtype=feat.dtype, device=feat.device)
        for idx, value in zip(max_indices[0], max_values[0]):
            if idx < logits.shape[1]:
                pseudo_label = logits[0, idx].argmax().item()
                x_query_select = feat[0, idx]
                weighted_x_query_select = value * x_query_select
                weighted_features[pseudo_label] += weighted_x_query_select / k
        proto = proto + weighted_features
        return proto

    def recon(self, proto, feat):
        proto = proto.squeeze(0)
        feat = feat.squeeze(0)

        norm_x = torch.norm(proto, dim=1, keepdim=True)
        norm_y = torch.norm(feat, dim=1, keepdim=True)
        x_norm = proto / norm_x
        y_norm = feat / norm_y

        similarity = torch.matmul(x_norm, y_norm.T)
        similarity_softmax = torch.softmax(similarity, dim=1)
        weighted_y_sum = similarity_softmax @ feat
        proto_new = self.alpha * proto + (1 - self.alpha) * weighted_y_sum

        similarity_transpose = similarity.T
        similarity_transpose_softmax = torch.softmax(similarity_transpose, dim=1)
        weighted_x_sum = similarity_transpose_softmax @ proto
        feat_new = self.alpha * feat + (1 - self.alpha) * weighted_x_sum

        proto = proto_new.unsqueeze(0)
        feat = feat_new.unsqueeze(0)
        return proto, feat

    def distance(self, proto, feat):
        if self.method == 'dot':
            proto = proto.mean(dim=-2)
            proto = F.normalize(proto, dim=-1)
            feat = F.normalize(feat, dim=-1)
            metric = 'dot'
        elif self.method == 'cos':
            proto = proto.mean(dim=-2)
            metric = 'cos'
        elif self.method == 'sqr':
            proto = proto.mean(dim=-2)
            metric = 'sqr'

        logits = self.compute_logits(feat, proto, metric=metric, temp=self.temp_proto)
        proto = self.proto_refine(logits, feat, proto)
        logits = self.compute_logits(feat, proto, metric=metric, temp=self.temp_proto)

        return logits.view(-1, self.args.way).unsqueeze(0)

    def calculate_adaptive_fusion_weights(self, c_r, c_f):
        denominator = c_r + c_f
        epsilon = 1e-8
        denominator = torch.clamp(denominator, min=epsilon)
        w_ri = c_r / denominator
        w_fi = c_f / denominator
        return w_ri.permute(1, 0), w_fi.permute(1, 0)

    def forward(self, x_shot, x_query, dct_shot, dct_query):
        probability_k = self.distance(x_shot, x_query)
        absolute_certainty, _ = torch.max(probability_k, dim=2)
        probability_k = F.softmax(probability_k, dim=-1)
        relative_certainty = torch.sum(probability_k * torch.log(probability_k + 1e-10), dim=2)

        probability_k_dct = self.distance(dct_shot, dct_query)
        absolute_certainty_dct, _ = torch.max(probability_k_dct, dim=2)
        probability_k_dct = F.softmax(probability_k_dct, dim=-1)
        relative_certainty_dct = torch.sum(probability_k_dct * torch.log(probability_k_dct + 1e-10), dim=2)

        diff_abso = absolute_certainty - absolute_certainty_dct + relative_certainty - relative_certainty_dct

        res_indices = torch.zeros(self.args.way * self.args.query, dtype=torch.bool)
        dct_indices = torch.zeros(self.args.way * self.args.query, dtype=torch.bool)
        for i in range(self.args.way * self.args.query):
            if diff_abso[0, i] > 0:
                res_indices[i] = True
            elif diff_abso[0, i] < 0:
                dct_indices[i] = True
        res_indices = torch.where(res_indices)[0]
        dct_indices = torch.where(dct_indices)[0]
        if res_indices.numel() == 0 and dct_indices.numel() == 0:
            distillation_loss_dctl = 0
            distillation_loss_resl = 0
        elif res_indices.numel() != 0 and dct_indices.numel() == 0:
            logits_res_h = self.distance(x_shot, x_query)
            logits_dct_l = self.distance(dct_shot, dct_query)
            distillation_loss_dctl = distillation_loss(logits_dct_l, logits_res_h, absolute_certainty)
            distillation_loss_resl = 0
        elif res_indices.numel() == 0 and dct_indices.numel() != 0:
            logits_dct_h = self.distance(dct_shot, dct_query)
            logits_res_l = self.distance(x_shot, x_query)
            distillation_loss_resl = distillation_loss(logits_res_l, logits_dct_h, absolute_certainty_dct)
            distillation_loss_dctl = 0
        else:
            x_query_h = x_query[:, res_indices]
            dct_query_l = dct_query[:, res_indices]
            dct_query_h = dct_query[:, dct_indices]
            x_query_l = x_query[:, dct_indices]
            logits_res_h = self.distance(x_shot, x_query_h)
            logits_dct_l = self.distance(dct_shot, dct_query_l)
            logits_dct_h = self.distance(dct_shot, dct_query_h)
            logits_res_l = self.distance(x_shot, x_query_l)

            abso_certainty = absolute_certainty[:, res_indices]
            abso_certainty_dct = absolute_certainty_dct[:, dct_indices]
            distillation_loss_dctl = distillation_loss(logits_dct_l, logits_res_h, abso_certainty)
            distillation_loss_resl = distillation_loss(logits_res_l, logits_dct_h, abso_certainty_dct)

        w_r, w_f = self.calculate_adaptive_fusion_weights(absolute_certainty, absolute_certainty_dct)

        return self.temp * distillation_loss_resl, self.temp_dct * distillation_loss_dctl, res_indices, dct_indices, w_r, w_f
