import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class MHAtt(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(MHAtt, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / 8)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask=None):
        b, n, s = q.shape

        v = self.linear_v(v).view(b, -1, self.head, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(b, -1, self.head, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(b, -1, self.head, self.head_size).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(b, -1, self.hidden_dim)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -65504.0)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


class MHAtt_dct(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(MHAtt_dct, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / 8)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask=None):
        b, n, s = q.shape

        v = self.linear_v(v).view(b, -1, self.head, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(b, -1, self.head, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(b, -1, self.head, self.head_size).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(b, -1, self.hidden_dim)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -65504.0)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


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


class Encoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Encoder, self).__init__()

        self.mhatt = MHAtt(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


class Encoder_dct(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Encoder_dct, self).__init__()

        self.mhatt = MHAtt_dct(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN_dct(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Decoder, self).__init__()

        self.mhatt1 = MHAtt(hidden_dim, dropout_r, head)
        self.mhatt2 = MHAtt(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))

        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x


class Decoder_dct(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Decoder_dct, self).__init__()

        self.mhatt1 = MHAtt_dct(hidden_dim, dropout_r, head)
        self.mhatt2 = MHAtt_dct(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN_dct(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))

        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8, avg_pool=False):
        super(Transformer, self).__init__()
        self.enc_list = nn.ModuleList([Encoder(hidden_dim, dropout_r, head) for _ in range(1)])
        self.dec_list = nn.ModuleList([Decoder(hidden_dim, dropout_r, head) for _ in range(1)])
        self.enc_list_dct = nn.ModuleList([Encoder_dct(hidden_dim, dropout_r, head) for _ in range(1)])
        self.dec_list_dct = nn.ModuleList([Decoder_dct(hidden_dim, dropout_r, head) for _ in range(1)])
        self.avg_pool = avg_pool
        if self.avg_pool:
            self.img_avgpool = nn.AdaptiveAvgPool1d(1)
            self.que_avgpool = nn.AdaptiveAvgPool1d(1)
            self.dct_avgpool = nn.AdaptiveAvgPool1d(1)
            self.que_dct_avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, img, que, dct, que_dct, img_mask, que_mask, dct_mask, que_mask_dct):
        for enc in self.enc_list:
            que = enc(que, que_mask)

        for enc in self.enc_list_dct:
            que_dct = enc(que_dct, que_mask_dct)
        b, n, c = img.shape
        for dec in self.dec_list:
            img = dec(img, que, img_mask, que_mask)

        for dec in self.dec_list_dct:
            dct = dec(dct, que_dct, dct_mask, que_mask_dct)

        if self.avg_pool:
            img = self.img_avgpool(img.permute(0, 2, 1)).view(b, -1)
            que = self.que_avgpool(que.permute(0, 2, 1)).view(b, -1)
            dct = self.dct_avgpool(dct.permute(0, 2, 1)).view(b, -1)
            que_dct = self.que_dct_avgpool(que_dct.permute(0, 2, 1)).view(b, -1)

        return img, que, dct, que_dct

