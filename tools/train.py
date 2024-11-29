import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')




def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DE.CONTROLLER = args.controller

    cfg.freeze()
    default_setup(cfg, args)
    return cfg



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = 'configs/airship/vits_shot5.yaml'
    args.num_gpus = 1
    args.opts=[
        'MODEL.WEIGHTS', 'weights/trained/few-shot/vits_0089999.pth',
        'SOLVER.IMS_PER_BATCH', '1',
        'DE.OFFLINE_RPN_CONFIG', 'configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml',
        'OUTPUT_DIR', 'output/vits/airship_5shot/'
        ]
    cfg = setup(args)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()