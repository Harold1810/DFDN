import torch

from torch.utils.data import DataLoader

# from dataloader.fsl_vqa_base import FSLVQA as Dataset
from dataloader.fsl_vqa import FSLVQA as Dataset
from dataloader.samplers import CategoriesSampler_metaVQA


def get_dataloader(args):
    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers = 8

    trainset = Dataset('train', args, augment=args.augment)
    valset = Dataset('val', args, token_to_ix=trainset.token_to_ix)
    testset = Dataset('test', args, token_to_ix=trainset.token_to_ix)

    args.num_class = trainset.num_class
    args.pretrained_emb = trainset.pretrained_emb
    args.token_size = trainset.token_size

    train_sampler = CategoriesSampler_metaVQA(trainset.label2ind,
                                              num_episodes,
                                              args.way,
                                              args.shot + args.query + args.unlabeled,
                                              args.batch)
    train_loader = DataLoader(dataset=trainset,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)

    val_sampler = CategoriesSampler_metaVQA(valset.label2ind,
                                            args.num_eval_episodes,
                                            args.eval_way,
                                            args.eval_shot + args.eval_query + args.eval_unlabeled,
                                            args.batch)
    val_loader = DataLoader(dataset=valset,
                            num_workers=args.num_workers,
                            batch_sampler=val_sampler,
                            pin_memory=True)

    test_sampler = CategoriesSampler_metaVQA(testset.label2ind,
                                             int(1 / args.batch),
                                             args.eval_way,
                                             args.eval_shot + args.eval_query + args.eval_unlabeled,
                                             args.batch)
    test_loader = DataLoader(dataset=testset,
                             batch_sampler=test_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def get_dataloader_fpait(args):
    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers = 8

    trainset = Dataset('train', args, augment=args.augment)
    valset = Dataset('test', args, token_to_ix=trainset.token_to_ix)
    testset = Dataset('test', args, token_to_ix=trainset.token_to_ix)

    args.num_class = trainset.num_class
    args.pretrained_emb = trainset.pretrained_emb
    args.token_size = trainset.token_size

    train_sampler = CategoriesSampler_metaVQA(trainset.label2ind,
                                              num_episodes,
                                              args.way,
                                              args.shot + args.query + args.unlabeled,
                                              args.batch)
    train_loader = DataLoader(dataset=trainset,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)

    val_sampler = CategoriesSampler_metaVQA(valset.label2ind,
                                            int(200 / args.batch),
                                            args.eval_way,
                                            args.eval_shot + args.eval_query + args.eval_unlabeled,
                                            args.batch)
    val_loader = DataLoader(dataset=valset,
                            num_workers=args.num_workers,
                            batch_sampler=val_sampler,
                            pin_memory=True)

    test_sampler = CategoriesSampler_metaVQA(testset.label2ind,
                                             int(10000 / args.batch),
                                             args.eval_way,
                                             args.eval_shot + args.eval_query + args.eval_unlabeled,
                                             args.batch)
    test_loader = DataLoader(dataset=testset,
                             batch_sampler=test_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
