import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch import nn
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.evaluation import instances_to_coco_json



import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from contextlib import ExitStack

import cv2

from detectron2.evaluation import  inference_context



# class_names = ['sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse']
class_names = ['airship']


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


def get_color_map(i):
    return [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
        (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ][i]



def test(cfg, model, dataset):
    data_loader = DefaultTrainer.build_test_loader(cfg, dataset)
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        for idx, inputs in enumerate(data_loader):
            image = cv2.imread(inputs[0]['file_name'])
            targets = []
            output = model(inputs)[0]
            if "instances" in output:
                instances = instances_to_coco_json(output["instances"].to(torch.device("cpu")), inputs[0]["image_id"])
                for ins in instances:
                    bbox = [ins['bbox'][0], ins['bbox'][1], ins['bbox'][0]+ins['bbox'][2], ins['bbox'][1]+ins['bbox'][3]]
                    cate_id = ins['category_id']
                    score = ins['score']
                    if score < 0.5: continue
                    targets.append({'bbox': [int(v) for v in bbox], 'category_id': cate_id, 'score': score})
            print('##################################################')
            for target in targets:
                colormap = get_color_map(target['category_id'])
                print(class_names[target['category_id']])
                cv2.rectangle(image, (target['bbox'][0], target['bbox'][1]), (target['bbox'][2], target['bbox'][3]), colormap, 2)
                cv2.putText(image, class_names[target['category_id']], (target['bbox'][0], target['bbox'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colormap, 1)
            cv2.imshow('result', image)
            if cv2.waitKey(0) == 27:
                break






if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = 'configs/airship/vits_shot5.yaml'
    args.num_gpus = 1
    args.opts=[
        'SOLVER.IMS_PER_BATCH', '1',
        'DE.OFFLINE_RPN_CONFIG', 'configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml',
    ]
    print(args)
    model_path = 'output/vits/airship_5shot/model_final.pth'
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(model_path)
    test(cfg, model, 'airship_test')
