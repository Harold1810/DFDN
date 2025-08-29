import torch
import time
import torch.nn.functional as F
import os.path as osp
import numpy as np
import logging

from tqdm import tqdm
from torch.cuda.amp import autocast as autocast

from model.trainer.basetrainer import Trainer
from dataloader.utils_dataloader import get_dataloader
from model.trainer.utils_trainer import prepare_model, prepare_optimizer
from model.helper.utils_helper import (
    Averager, count_acc, compute_confidence_interval,
)


class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def make_nk_label(self, n, k, batch=1):
        label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
        label = label.repeat(batch)
        return label

    def train(self):
        args = self.args
        support_label = self.make_nk_label(args.way, args.shot, args.batch).cuda()
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()
        label_aux = torch.cat((support_label, label), dim=0)

        tl1 = Averager()

        if not args.pretrain_path:
            print("no pretrain")
        else:
            print("pretrain")
            model_dict = self.model.state_dict()
            pretrain_dict = torch.load(osp.join(self.args.pretrain_path, 'max_acc_sim.pth'))
            pretrain_dict = pretrain_dict['params']

            prefix_to_remove = "encoder."
            new_pretrain_dict = {}
            for key, value in pretrain_dict.items():
                if key.startswith(prefix_to_remove):
                    new_key = key[len(prefix_to_remove):]
                    new_pretrain_dict[new_key] = value
                else:
                    new_pretrain_dict[key] = value
            prefixes_to_exclude = ["temp", "temp_dct"]
            # prefixes_to_exclude = ["encoder.", "dct_encoder.", "temp", "temp_dct"]
            pretrain_dict = {key: value for key, value in new_pretrain_dict.items() if
                             not any(key.startswith(prefix) for prefix in prefixes_to_exclude)}
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(model_dict)

        for epoch in range(1, args.max_epoch + 1):

            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()

            tl2 = Averager()
            ta = Averager()
            tar = Averager()
            tad = Averager()

            start_tm = time.time()
            train_gen = tqdm(self.train_loader)

            for i, batch in enumerate(train_gen, 1):
                self.train_step += 1
                if torch.cuda.is_available():
                    data, que, gt_label, dct = [_.cuda() for _ in batch]
                else:
                    data, que, gt_label, dct = batch[0], batch[1], batch[2], batch[3]
                data_tm = time.time()
                with autocast():

                    logits_res, logits_dct, distillation_loss_resl, distillation_loss_dctl, res_indices, dct_indices, logits, recon_loss, recon_loss_dct, attention_difference_loss, attention_difference_loss_dct = self.model(
                        data, que, support_label, dct)
                    entropy_res = F.cross_entropy(logits_res, label)
                    entropy_dct = F.cross_entropy(logits_dct, label)
                    entropy = F.cross_entropy(logits, label)
                    loss = entropy_res + entropy_dct + distillation_loss_resl + distillation_loss_dctl + recon_loss + recon_loss_dct + entropy - attention_difference_loss - attention_difference_loss_dct

                tl2.add(loss)
                tl1.add(loss.item())
                forward_tm = time.time()
                acc = count_acc(logits, label)
                acc_res = count_acc(logits_res, label)
                acc_dct = count_acc(logits_dct, label)
                ta.add(acc)
                tar.add(acc_res)
                tad.add(acc_dct)
                train_gen.set_description(
                    '训练阶段:epo {} total_loss={:.4f} partial_loss={:.4f} 平均acc={:.4f} 平均acc_res={:.4f} 平均acc_dct={:.4f}'.format(epoch, tl1.item(), tl2.item(), ta.item(), tar.item(), tad.item()))
                logging.info(train_gen)
                self.optimizer.zero_grad()
                loss.backward()
                backward_tm = time.time()
                self.optimizer.step()
                optimizer_tm = time.time()

                self.dt.add(data_tm - start_tm)
                self.ft.add(forward_tm - data_tm)
                self.bt.add(backward_tm - forward_tm)
                self.ot.add(optimizer_tm - backward_tm)
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print('ETA:{}/{}'.format(
                self.timer.measure(),
                self.timer.measure(self.train_epoch / args.max_epoch))
            )
        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        args = self.args
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))
        support_label = self.make_nk_label(args.way, args.shot, args.batch).cuda()
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        with torch.no_grad():
            val_gen = tqdm(data_loader)
            tl1 = Averager()
            ta = Averager()
            tar = Averager()
            tad = Averager()
            for i, batch in enumerate(val_gen, 1):
                if torch.cuda.is_available():
                    data, que, _, dct = [_.cuda() for _ in batch]
                else:
                    data, que, dct = batch[0], batch[1], batch[3]

                logits_res, logits_dct, distillation_loss_resl, distillation_loss_dctl, res_indices, dct_indices, logits, recon_loss, recon_loss_dct, attention_difference_loss, attention_difference_loss_dct = self.model(
                    data, que, support_label, dct)
                entropy_res = F.cross_entropy(logits_res, label)
                entropy_dct = F.cross_entropy(logits_dct, label)
                entropy = F.cross_entropy(logits, label)
                loss = entropy_res + entropy_dct + distillation_loss_resl + distillation_loss_dctl + recon_loss + recon_loss_dct + entropy - attention_difference_loss - attention_difference_loss_dct

                acc = count_acc(logits, label)
                acc_res = count_acc(logits_res, label)
                acc_dct = count_acc(logits_dct, label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc
                tl1.add(loss)
                ta.add(acc)
                tar.add(acc_res)
                tad.add(acc_dct)
                val_gen.set_description('验证阶段:平均loss={:.4f} 平均acc={:.4f} 平均acc_res={:.4f} 平均acc_dct={:.4f}'.format(tl1.item(), ta.item(), tar.item(), tad.item()))
                logging.info(val_gen)
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        return vl, va, vap

    def evaluate_test(self):
        args = self.args
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((int(10000 / self.args.batch), 2))
        support_label = self.make_nk_label(args.way, args.shot, args.batch).cuda()
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        with torch.no_grad():
            tl1 = Averager()
            ta = Averager()
            tar = Averager()
            tad = Averager()
            test_gen = tqdm(self.test_loader)
            for i, batch in enumerate(test_gen, 1):
                if torch.cuda.is_available():
                    data, que, _, dct = [_.cuda() for _ in batch]
                else:
                    data, que, dct = batch[0], batch[1], batch[3]

                logits_res, logits_dct, distillation_loss_resl, distillation_loss_dctl, res_indices, dct_indices, logits, recon_loss, recon_loss_dct, attention_difference_loss, attention_difference_loss_dct = self.model(
                    data, que, support_label, dct)
                entropy_res = F.cross_entropy(logits_res, label)
                entropy_dct = F.cross_entropy(logits_dct, label)
                entropy = F.cross_entropy(logits, label)
                loss = entropy_res + entropy_dct + distillation_loss_resl + distillation_loss_dctl + recon_loss + recon_loss_dct + entropy - attention_difference_loss - attention_difference_loss_dct
                acc = count_acc(logits, label)
                acc_res = count_acc(logits_res, label)
                acc_dct = count_acc(logits_dct, label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc
                tl1.add(loss)
                ta.add(acc)
                tar.add(acc_res)
                tad.add(acc_dct)
                test_gen.set_description('测试阶段:平均loss1={:.4f} 平均acc={:.4f} 平均acc_res={:.4f} 平均acc_dct={:.4f}'.format(tl1.item(), ta.item(), tar.item(), tad.item()))
                logging.info(test_gen)
        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl
        torch.save(self.model.state_dict(), args.save_path + '_{}.pth'.format(va))
        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
            self.trlog['test_acc'],
            self.trlog['test_acc_interval']))
        return vl, va, vap

    def final_record(self):
        with open(
                osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])),
                'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))
