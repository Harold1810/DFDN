import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMaskGenerator(nn.Module):
    def __init__(self, num_classes, temp=1.0):
        super(AttentionMaskGenerator, self).__init__()
        self.num_classes = num_classes
        self.temp = temp

    def forward(self, features, support, query):
        M = features
        scores = self.compute_cosine_similarity(support, query)
        attention_maps = self.generate_attention_map(M, support, scores)

        masked_features, unmasked_features = self.apply_mask(attention_maps, M)
        return attention_maps, scores, masked_features, unmasked_features

    def generate_attention_map(self, M, support, scores):
        attention_maps = []
        M.requires_grad_(True)
        support.requires_grad_(True)
        scores.requires_grad_(True)
        with torch.enable_grad():
            for c in range(self.num_classes):
                score_c = scores.mean(dim=0)[c]
                grad = torch.autograd.grad(score_c, M, retain_graph=True, create_graph=True, allow_unused=True)[0]
                if grad is None:
                    grad = torch.zeros_like(support)
                w_c_k = grad.mean(dim=0)
                A_c = F.relu(w_c_k.view(1, -1) * M)
                attention_maps.append(A_c)
        return torch.stack(attention_maps)

    def compute_cosine_similarity(self, support, query):
        class_prototypes = support.mean(dim=0)
        scores = F.cosine_similarity(query.unsqueeze(1), class_prototypes.unsqueeze(0), dim=1)
        return scores

    def apply_mask(self, attention_maps, D):
        masked_features = torch.zeros_like(D)
        unmasked_features = torch.zeros_like(D)
        for A_c in attention_maps:
            Mask_c = torch.sigmoid(A_c)
            masked_features += Mask_c * D
            unmasked_features += (1 - Mask_c) * D
        return masked_features, unmasked_features
