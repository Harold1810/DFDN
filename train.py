import os
import numpy as np
import torch
import random
import time
import argparse
import pprint

from utils import set_pretrain

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_init_weight(args):
    init_path = {'Res12': './initialization/Res12-Pre/res12_checkpoint.pth',
                 'SwinT': './initialization/SwinT-Pre/swint_checkpoint.pth',
                 'VitS': './initialization/Vit-Pre/vit_checkpoint.pth'}
    init_dct_path = {'Res12': './initialization/Res12-Pre/res12_dct_checkpoint.pth',
                     'SwinT': './initialization/SwinT-Pre/swint_dct_checkpoint.pth',
                     'VitS': './initialization/Vit-Pre/vit_dct_checkpoint.pth'}
    args.init_weights = init_path[args.backbone_class]
    args.init_dct_weights = init_dct_path[args.backbone_class]


def set_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def postprocess_args(args):
    set_init_weight(args)
    set_gpu(args.gpu)
    set_seed(args.seed)
    set_pretrain(args)
    save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class,
                           '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
    save_path2 = '-'.join([str(time.strftime('%Y%m%d_%H%M%S'))])

    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.makedirs(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=200)

    # model
    parser.add_argument('--model_class', type=str, default='ProtoNet',
                        choices=['ProtoNet', 'ProtoNetTrue'])
    parser.add_argument('--backbone_class', type=str, default='Res12',
                        choices=['Res12', 'SwinT', 'VitS'])
    parser.add_argument('--dataset', type=str, default='COCO',
                        choices=['COCO', 'VG_QA', 'VQAv2'])
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--eval_shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--eval_query', type=int, default=5)
    parser.add_argument('--unlabeled', type=int, default=0)
    parser.add_argument('--eval_unlabeled', type=int, default=0)
    parser.add_argument('--batch', type=int, default=1)

    # optimization parameters
    parser.add_argument('--lr', type=float, default=0.000025)
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--fix_BN', action='store_true', default=False)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--init_weights', type=str, default='./initialization/Res12-Pre/res12_checkpoint.pth')
    parser.add_argument('--init_dct_weights', type=str, default='./initialization/Res12-Pre/res12_checkpoint.pth')

    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default='31415926', help='random seed')
    parser.add_argument('--pretrain', default=False)
    parser.add_argument('--k_value', type=int, default=15)
    parser.add_argument('--use_fapit', type=int, default=0)
    return parser


def main():
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    pprint(vars(args))
    if args.use_fapit == 0:
        # from model.trainer.fsl.FSLTrainer_base import FSLTrainer
        from model.trainer.fsl.FSLTrainer import FSLTrainer
    else:
        from model.trainer.fsl.FSLTrainer_fpait import FSLTrainer
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    trainer.final_record()


if __name__ == '__main__':
    main()



