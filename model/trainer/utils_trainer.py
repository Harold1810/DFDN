import torch
import torch.nn as nn
import torch.optim as optim
from model.protonet import ProtoNet


def prepare_model(args):
    model = eval(args.model_class)(args)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_weights)['params']
        if args.backbone_class in ['SwinT', 'VitS']:
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model.to(device)
    return model, para_model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()}, {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
        )
    else:
        optimizer = optim.SGD(
            [{'params': model.encoder.parameters()}, {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay,
        )

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.max_epoch,
            eta_min=0
        )
    else:
        raise ValueError('No Such Scheduler')
    return optimizer, lr_scheduler
