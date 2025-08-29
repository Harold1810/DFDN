import torch.nn as nn
import torch
import torch.nn.functional as F

# from model.base.basemodel_base import FewShotModel
from model.base.basemodel import FewShotModel


class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.temp = nn.Parameter(torch.tensor(10., requires_grad=True))
        self.method = 'dot'
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

    def _forward(self, proto, feat):
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
        logits = self.compute_logits(feat, proto, metric=metric, temp=self.temp)
        proto = self.proto_refine(logits, feat, proto)
        logits = self.compute_logits(feat, proto, metric=metric, temp=self.temp)

        return logits.view(-1, self.args.way)
