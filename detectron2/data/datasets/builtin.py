# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annottions

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
from detectron2.data import MetadataCatalog
from .coco import register_coco_instances
import json
from detectron2.data.datasets import register_coco_instances

# custom datasets
dataset = 'airship'

_PREDEFINED_CD = []

_PREDEFINED_CD.append((f'{dataset}_train', f'{dataset}/train',f'{dataset}/annotations/train.json'))
_PREDEFINED_CD.append((f'{dataset}_test', f'{dataset}/test',f'{dataset}/annotations/test.json'))
for shot in [1, 5]:
    _PREDEFINED_CD.append((f'{dataset}_{shot}shot',f'{dataset}/train',f'{dataset}/annotations/{shot}_shot.json'))

def register_all_CD(root):
    for name, image_dir, json_file in _PREDEFINED_CD:
        with open(os.path.join(root, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = [i["name"] for i in data["categories"]]
        register_coco_instances(name, {}, os.path.join(root, json_file), os.path.join(root, image_dir))
        MetadataCatalog.get(name).set(thing_classes=classes)

# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # TODO replace to your own dataset path
    _CD_root = os.getenv("DETECTRON2_DATASETS", "datasets/")
    register_all_CD(_CD_root)
