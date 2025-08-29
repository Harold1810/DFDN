import torch.nn as nn
import torch
import torch.nn.functional as F

from model.networks.backbone.LSTM.lstm import LSTM
from model.networks.backbone.LSTM.lstm_dct import LSTM_dct
from model.networks.trans.transformer import Transformer
from model.networks.entropy.entropy import Entropy
from model.networks.distillation.distillation import Distillation


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class FewShotModel(nn.Module):
    def __init__(self, args, hidden_dim=768):
        super().__init__()
        self.args = args
        if args.backbone_class == 'Res12':
            from model.networks.backbone.ResNet.res12 import ResNet
            self.encoder = ResNet(avg_pool=False)
            from model.networks.backbone.ResNet.res12_dct import ResNet_dct
            self.dct_encoder = ResNet_dct(avg_pool=False)
            hidden_dim = 640
        elif args.backbone_class == 'SwinT':
            from model.networks.backbone.SwinT.swin_transformer import SwinTransformer
            self.encoder = SwinTransformer(window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                           mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1)
            from model.networks.backbone.SwinT.swin_transformer_dct import SwinTransformer as SwinTransformer_dct
            self.dct_encoder = SwinTransformer_dct(window_size=7, embed_dim=96, depths=[2, 2, 6, 2],
                                                   num_heads=[3, 6, 12, 24],
                                                   mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1)
            hidden_dim = 768
        elif args.backbone_class == 'VitS':
            from model.networks.backbone.vision_transformer import VisionTransformer
            self.encoder = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                             qkv_bias=True)
            hidden_dim = 384
        else:
            raise ValueError('')
        self.que_encoder = LSTM(args.pretrained_emb, args.token_size, hidden_dim=hidden_dim, avg_pool=False)
        self.que_encoder_dct = LSTM_dct(args.pretrained_emb, args.token_size, hidden_dim=hidden_dim, avg_pool=False)

        self.transformer = Transformer(hidden_dim=hidden_dim)
        self.entropy = Entropy(hidden_dim=hidden_dim)
        self.x_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dct_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        self.distillation_all = Distillation(self.args, hidden_dim=hidden_dim)
        self.temp = nn.Parameter(torch.tensor(1., requires_grad=True))
        self.temp_dct = nn.Parameter(torch.tensor(1., requires_grad=True))

    def split_shot_query(self, data, que, dct, ep_per_batch=1):
        args = self.args
        img_shape = data.shape[1:]
        data = data.view(ep_per_batch, args.way, args.shot + args.query, *img_shape)
        x_shot, x_query = data.split([args.shot, args.query], dim=2)
        x_shot = x_shot.contiguous()
        x_query = x_query.contiguous().view(ep_per_batch, args.way * args.query, *img_shape)

        que_shape = que.shape[1:]
        que = que.view(ep_per_batch, args.way, args.shot + args.query, *que_shape)
        que_shot, que_query = que.split([args.shot, args.query], dim=2)
        que_shot = que_shot.contiguous()
        que_query = que_query.contiguous().view(ep_per_batch, args.way * args.query, *que_shape)

        dct_shape = dct.shape[1:]
        dct = dct.view(ep_per_batch, args.way, args.shot + args.query, *dct_shape)
        dct_shot, dct_query = dct.split([args.shot, args.query], dim=2)
        dct_shot = dct_shot.contiguous()
        dct_query = dct_query.contiguous().view(ep_per_batch, args.way * args.query, *dct_shape)

        return x_shot, x_query, que_shot, que_query, dct_shot, dct_query

    def forward(self, x, que, support_labels, dct, get_feature=False):
        if get_feature:
            return self.encoder(x)
        else:
            x_shot, x_query, que_shot, que_query, dct_shot, dct_query = self.split_shot_query(x, que, dct, self.args.batch)
            shot_shape = x_shot.shape[:-3]
            query_shape = x_query.shape[:-3]

            img_shape = x_shot.shape[-3:]
            que_shape = que_shot.shape[-1:]
            dct_shape = dct_shot.shape[-3:]

            x_shot = x_shot.view(-1, *img_shape)
            x_query = x_query.view(-1, *img_shape)
            que_shot = que_shot.view(-1, *que_shape)
            que_query = que_query.view(-1, *que_shape)
            dct_shot = dct_shot.view(-1, *dct_shape)
            dct_query = dct_query.view(-1, *dct_shape)

            if self.args.backbone_class in ['VitS', 'SwinT']:
                x_tot = self.encoder.forward(torch.cat([x_shot, x_query], dim=0), return_all_tokens=True)[:, 1:]
                dct_tot = self.dct_encoder.forward(torch.cat([dct_shot, dct_query], dim=0), return_all_tokens=True)[:,
                          1:]
            else:
                x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
                dct_tot = self.dct_encoder(torch.cat([dct_shot, dct_query], dim=0))
            que_tot = self.que_encoder(torch.cat([que_shot, que_query], dim=0))
            que_tot_dct = self.que_encoder_dct(torch.cat([que_shot, que_query], dim=0))

            img_mask = make_mask(x_tot)
            que_mask = make_mask(que_tot)
            dct_mask = make_mask(dct_tot)
            que_mask_dct = make_mask(que_tot_dct)

            img_ori = x_tot
            dct_ori = dct_tot
            len_shot = len(x_shot)
            len_query = len(x_query)

            x_tot, que_tot, dct_tot, que_tot_dct = self.transformer(x_tot, que_tot, dct_tot, que_tot_dct, img_mask, que_mask, dct_mask, que_mask_dct)
            recon_loss, recon_loss_dct, x_tot, dct_tot, que_tot, que_tot_dct, attention_difference_loss, attention_difference_loss_dct = self.entropy(x_tot, que_tot, dct_tot, que_tot_dct, img_mask, que_mask, dct_mask, que_mask_dct, img_ori, dct_ori, len_shot, len_query)
            x_tot = self.x_linear(torch.cat([x_tot, que_tot], dim=-1))
            dct_tot = self.dct_linear(torch.cat([dct_tot, que_tot_dct], dim=-1))

            feat_shape = x_tot.shape[1:]
            x_shot, x_query = x_tot[:len_shot], x_tot[-len_query:]
            x_shot = x_shot.view(*shot_shape, *feat_shape)
            x_query = x_query.view(*query_shape, *feat_shape)

            dct_shot, dct_query = dct_tot[:len_shot], dct_tot[-len_query:]
            dct_shot = dct_shot.view(*shot_shape, *feat_shape)
            dct_query = dct_query.view(*query_shape, *feat_shape)

            distillation_loss_resl, distillation_loss_dctl, res_indices, dct_indices, w_r, w_f = self.distillation_all(x_shot, x_query, dct_shot, dct_query)
            logits_res = self._forward(x_shot, x_query)
            logits_dct = self._forward(dct_shot, dct_query)

            logits_final = self.temp * w_r * logits_res + self.temp_dct * w_f * logits_dct
            norm = logits_final.norm(p=2, dim=1, keepdim=True)
            logits_final = logits_final / (norm + 1e-6)
            # logits_final = F.normalize(logits_final, p=2, dim=1)
            return logits_res, logits_dct, distillation_loss_resl, distillation_loss_dctl, res_indices, dct_indices, logits_final, recon_loss, recon_loss_dct, attention_difference_loss, attention_difference_loss_dct

    def _forward(self, x_shot, x_query):
        raise NotImplementedError('Suppose to be implemented by subclass')
