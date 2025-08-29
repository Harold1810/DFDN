import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from scipy.fftpack import dct, idct

from model.networks.entropy.mask import AttentionMaskGenerator
from model.networks.entropy.mask_dct import AttentionMaskGenerator as AttentionMaskGenerator_dct


def apply_idct(dct_tot):
    dct_tot_np = dct_tot.detach().cpu().numpy()
    idct_result_np = idct(dct_tot_np, norm='ortho', axis=-1)
    idct_result = torch.from_numpy(idct_result_np).to(dct_tot.device)
    return idct_result


def apply_dct(x_tot):
    x_tot_np = x_tot.detach().cpu().numpy()
    dct_result_np = dct(x_tot_np, norm='ortho', axis=-1)
    dct_result = torch.from_numpy(dct_result_np).to(x_tot.device)
    return dct_result


class BiAttention(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1):
        super(BiAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.l_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim)
        self.i_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim)
        self.final = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, i_batch, q_batch, i_mask, q_mask):
        i_feat = self.qkv_attention(i_batch, q_batch, q_batch, q_mask)
        i_feat, i_weight = self.l_flatten(i_feat, i_mask)
        l_feat = self.qkv_attention(q_batch, i_batch, i_batch, i_mask)
        l_feat, _ = self.i_flatten(l_feat, q_mask)
        return i_feat, l_feat, i_weight

    def qkv_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.data.masked_fill_(mask.squeeze(1), -65504.0)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)


class BiAttention_dct(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1):
        super(BiAttention_dct, self).__init__()
        self.hidden_dim = hidden_dim
        self.l_flatten = AttFlat_dct(hidden_dim, dropout_r, hidden_dim)
        self.i_flatten = AttFlat_dct(hidden_dim, dropout_r, hidden_dim)
        self.final = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, i_batch, q_batch, i_mask, q_mask):
        i_feat = self.qkv_attention(i_batch, q_batch, q_batch, q_mask)
        i_feat, i_weight = self.l_flatten(i_feat, i_mask)
        l_feat = self.qkv_attention(q_batch, i_batch, i_batch, i_mask)
        l_feat, _ = self.i_flatten(l_feat, q_mask)
        return i_feat, l_feat, i_weight

    def qkv_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.data.masked_fill_(mask.squeeze(1), -65504.0)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)


class AGAttention(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1):
        super(AGAttention, self).__init__()
        self.lin_v = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)
        self.lin_q = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)
        self.lin = PositionWiseFFN(hidden_dim, dropout_r, 1)

    def forward(self, v, q, v_mask):
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)
        x = v * q
        x = self.lin(x)
        x = x.squeeze(-1).masked_fill(v_mask.squeeze(2).squeeze(1), -65504.0)
        x = F.softmax(x, dim=1)
        return x


class AGAttention_dct(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1):
        super(AGAttention_dct, self).__init__()
        self.lin_v = PositionWiseFFN_dct(hidden_dim, dropout_r, hidden_dim)
        self.lin_q = PositionWiseFFN_dct(hidden_dim, dropout_r, hidden_dim)
        self.lin = PositionWiseFFN_dct(hidden_dim, dropout_r, 1)

    def forward(self, v, q, v_mask):
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)
        x = v * q
        x = self.lin(x)
        x = x.squeeze(-1).masked_fill(v_mask.squeeze(2).squeeze(1), -65504.0)
        x = F.softmax(x, dim=1)
        return x


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, outdim=640):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_r)
        self.dense2 = nn.Linear(hidden_dim * 2, outdim)

    def forward(self, X):
        return self.dense2(self.dropout(self.relu(self.dense1(X))))


class PositionWiseFFN_dct(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, outdim=640):
        super(PositionWiseFFN_dct, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_r)
        self.dense2 = nn.Linear(hidden_dim * 2, outdim)

    def forward(self, X):
        return self.dense2(self.dropout(self.relu(self.dense1(X))))


class AttFlat(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, out_dim=640, glimpses=1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses
        self.mlp = PositionWiseFFN(hidden_dim, dropout_r, self.glimpses)
        self.linear_merge = nn.Linear(hidden_dim * glimpses, out_dim)

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -65504.0
            )
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted, att.squeeze()


class AttFlat_dct(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, out_dim=640, glimpses=1):
        super(AttFlat_dct, self).__init__()
        self.glimpses = glimpses
        self.mlp = PositionWiseFFN_dct(hidden_dim, dropout_r, self.glimpses)
        self.linear_merge = nn.Linear(hidden_dim * glimpses, out_dim)

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -65504.0
            )
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted, att.squeeze()


class Entropy(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1):
        super(Entropy, self).__init__()
        self.bi_attention = BiAttention(hidden_dim, dropout_r)
        self.bi_attention_dct = BiAttention_dct(hidden_dim, dropout_r)
        self.attflat = AttFlat(hidden_dim, dropout_r, hidden_dim * 2)
        self.attflat_dct = AttFlat_dct(hidden_dim, dropout_r, hidden_dim * 2)
        self.attflat_que = AttFlat(hidden_dim, dropout_r, hidden_dim * 2)
        self.attflat_que_dct = AttFlat_dct(hidden_dim, dropout_r, hidden_dim * 2)
        self.ag_attention = AGAttention(hidden_dim, dropout_r)
        self.ag_attention_dct = AGAttention_dct(hidden_dim, dropout_r)
        self.linear_que = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_que_dct = nn.Linear(hidden_dim * 2, hidden_dim)
        self.temp = nn.Parameter(torch.tensor(100., requires_grad=True))
        self.temp_dct = nn.Parameter(torch.tensor(100., requires_grad=True))
        self.mask = nn.Parameter(torch.tensor(100., requires_grad=True))
        self.mask_dct = nn.Parameter(torch.tensor(100., requires_grad=True))
        self.generator = AttentionMaskGenerator(num_classes=5)
        self.generator_dct = AttentionMaskGenerator_dct(num_classes=5)

    def _recon_loss_kl(self, attn_weight, recon_weight, temp):
        attn_weight = F.softmax(attn_weight, dim=-1)
        recon_weight = F.softmax(recon_weight, dim=-1)
        kl_div = F.kl_div(attn_weight.log(), recon_weight, reduction='sum')
        weighted_kl_div = kl_div
        return weighted_kl_div * temp

    def recon_loss_enhance(self, attn_weight, recon_weight):
        return self._recon_loss_kl(attn_weight, recon_weight, self.temp)

    def recon_loss_enhance_dct(self, attn_weight, recon_weight):
        return self._recon_loss_kl(attn_weight, recon_weight, self.temp_dct)

    def compute_attention_difference_loss(self, attn_weight, mask):
        return F.mse_loss(attn_weight, mask)

    def forward(self, img, que, dct, que_dct, img_mask, que_mask, dct_mask, que_mask_dct, img_ori, dct_ori, len_shot, len_query):
        self.total_num = img_ori.shape[1]
        x_tot, que_tot, attn_weight = self.bi_attention(img, que, img_mask, que_mask)
        dct_tot, que_tot_dct, attn_weight_dct = self.bi_attention_dct(dct, que_dct, dct_mask, que_mask_dct)

        x_shot, x_query = x_tot[:len_shot], x_tot[-len_query:]
        attention_maps, scores, masked_x, unmasked_x = self.generator(x_tot, x_shot, x_query)
        dct_shot, dct_query = dct_tot[:len_shot], dct_tot[-len_query:]
        attention_maps, scores, masked_dct, unmasked_dct = self.generator_dct(dct_tot, dct_shot, dct_query)

        b, n, c = img_ori.shape
        unmask_idct = apply_idct(unmasked_dct).unsqueeze(2).permute(0, 2, 1).expand(b, n, c)
        unmask_dct = apply_dct(unmasked_x).unsqueeze(2).permute(0, 2, 1).expand(b, n, c)
        mask_idct = apply_idct(masked_dct).unsqueeze(2).permute(0, 2, 1).expand(b, n, c)
        mask_dct = apply_dct(masked_x).unsqueeze(2).permute(0, 2, 1).expand(b, n, c)

        que_res, que_weight_res = self.attflat_que(que, que_mask)
        que_dct, que_weight_dct = self.attflat_que_dct(que_dct, que_mask_dct)

        _, idct_weight = self.attflat(unmask_idct, img_mask)
        _, dct_weight = self.attflat_dct(unmask_dct, dct_mask)
        _, mask_idct_weight = self.attflat(mask_idct, img_mask)
        _, mask_dct_weight = self.attflat_dct(mask_dct, dct_mask)

        recon_weight = self.ag_attention(img_ori, self.linear_que(que_res), img_mask) + idct_weight
        recon_loss = self.recon_loss_enhance(attn_weight=attn_weight, recon_weight=recon_weight)

        recon_weight_dct = self.ag_attention_dct(dct_ori, self.linear_que_dct(que_dct), dct_mask) + dct_weight
        recon_loss_dct = self.recon_loss_enhance_dct(attn_weight=attn_weight_dct, recon_weight=recon_weight_dct)

        attention_difference_loss = self.compute_attention_difference_loss(attn_weight, mask_idct_weight)
        attention_difference_loss_dct = self.compute_attention_difference_loss(attn_weight_dct, mask_dct_weight)

        return recon_loss, recon_loss_dct, x_tot, dct_tot, que_tot, que_tot_dct, self.mask * attention_difference_loss, self.mask_dct * attention_difference_loss_dct
