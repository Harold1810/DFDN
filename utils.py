
def set_pretrain(args):
    pretrain_paths = {
        'COCO': {
            'Res12': {
                '5': {
                    '0': './pre-model/COCO-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/COCO-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 110]/',
                },
                '10': {
                    '0': './pre-model/COCO-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/COCO-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 115/',
                }
            },
            'SwinT': {
                '5': {
                    '0': './pre-model/COCO-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/COCO-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 110]/',
                },
                '10': {
                    '0': './pre-model/COCO-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/COCO-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 115]/',
                }
            },
            'VitS': {
                '5': {
                    '0': './pre-model/COCO-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/COCO-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 110]/',
                },
                '10': {
                    '0': './pre-model/COCO-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/COCO-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 115]/',
                }
            }
        },
        'VQAv2': {
            'Res12': {
                '5': {
                    '0': './pre-model/VQAv2-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/VQAv2-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                },
                '10': {
                    '0': './pre-model/VQAv2-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/VQAv2-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                }
            },
            'SwinT': {
                '5': {
                    '0': './pre-model/VQAv2-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/VQAv2-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                },
                '10': {
                    '0': './pre-model/VQAv2-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/VQAv2-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                }
            },
            'VitS': {
                '5': {
                    '0': './pre-model/VQAv2-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/VQAv2-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                },
                '10': {
                    '0': './pre-model/VQAv2-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/VQAv2-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                }
            }
        },
        'VG_QA': {
            'Res12': {
                '5': {
                    '0': './pre-model/VG_QA-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/VG_QA-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 110]/',
                },
                '10': {
                    '0': './pre-model/VG_QA-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/VG_QA-Res12-Pre-idea/2.5e-05_0.1_[25, 50, 115]/',
                }
            },
            'SwinT': {
                '5': {
                    '0': './pre-model/VG_QA-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/VG_QA-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 110]/',
                },
                '10': {
                    '0': './pre-model/VG_QA-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/VG_QA-SwinT-Pre-idea/2.5e-05_0.1_[25, 50, 115]/',
                }
            },
            'VitS': {
                '5': {
                    '0': './pre-model/VG_QA-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 100]/',
                    '1': './pre-model/VG_QA-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 110]/',
                },
                '10': {
                    '0': './pre-model/VG_QA-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 105]/',
                    '1': './pre-model/VG_QA-VitS-Pre-idea/2.5e-05_0.1_[25, 50, 115]/',
                }
            }
        }
    }

    if args.pretrain:
        if args.dataset in pretrain_paths and args.backbone_class in pretrain_paths[args.dataset]:
            args.pretrain_path = pretrain_paths[args.dataset][args.backbone_class][str(args.way)][str(args.use_fapit)]
        else:
            raise ValueError("Invalid dataset or backbone class.")
    else:
        args.pretrain_path = None
