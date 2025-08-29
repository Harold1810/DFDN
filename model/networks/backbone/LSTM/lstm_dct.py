import torch.nn as nn
import torch


class LSTM_dct(nn.Module):
    def __init__(self, pretrained_emb, token_size, hidden_dim=640, avg_pool=True, emb_size=300):
        super(LSTM_dct, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=emb_size
        )

        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.avg_pool = avg_pool
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, all_que):
        lang_feat = self.embedding(all_que)
        lang_feat, _ = self.lstm(lang_feat)
        if self.avg_pool:
            lang_feat = self.avgpool(lang_feat.permute(0, 2, 1)).view(lang_feat.size(0), -1)
        return lang_feat
