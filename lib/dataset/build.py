# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Congju Du (ducongju@hust.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data
import numpy as np
from .COCODataset import CocoDataset as coco
from .COCODatasetGetScoreData import CocoDatasetGetScoreData as cocoscore
from .CrowdPoseDatasetGetScoreData import CrowdPoseDatasetGetScoreData as crowdposescore
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import OffsetGenerator
from .target_generators import ScaleAwareHeatmapGenerator


def build_dataset(cfg, is_train):
    # is_train判断是否训练，只有在训练时候才会对数据进行转换处理
    transforms = build_transforms(cfg, is_train)

    ################################################################
    if cfg.DATASET.SCALE_AWARE_SIGMA:
        heatmap_generator = [
            ScaleAwareHeatmapGenerator(
                output_size, cfg.DATASET.NUM_JOINTS, 
                cfg.DATASET.USE_JNT, cfg.DATASET.JNT_THR, cfg.DATASET.USE_INT,
                cfg.DATASET.SHAPE, cfg.DATASET.SHAPE_WEIGHT, cfg.DATASET.PAUTA
            ) for output_size in cfg.DATASET.OUTPUT_SIZE
        ]
    else:
        heatmap_generator = [
            HeatmapGenerator(
                output_size, cfg.DATASET.NUM_JOINTS
            ) for output_size in cfg.DATASET.OUTPUT_SIZE
        ]
    ################################################################

    offset_generator = None
    if cfg.DATASET.OFFSET_REG:
        offset_generator = [
            OffsetGenerator(
                output_size,
                output_size,
                cfg.DATASET.NUM_JOINTS,
                cfg.DATASET.OFFSET_RADIUS
            ) for output_size in cfg.DATASET.OUTPUT_SIZE
        ]

    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST

    # eval()函数用来执行一个字符串表达式，并返回表达式的值
    # 在这里eval(cfg.DATASET.DATASET)返回的是数据集的名字coco_kpt或者crowd_pose_kpt
    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        dataset_name,
        is_train,
        heatmap_generator,
        offset_generator,
        transforms
    )

    ################################################################
    if cfg.DATASET.USE_SUBSET:
        validation_split = cfg.DATASET.SUBSET_FACTOR
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        _, val_indices = indices[split:], indices[:split]
        valid_dataset = torch.utils.data.Subset(dataset, val_indices)
        dataset = valid_dataset
    ################################################################

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, is_train)

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    # 生成测试集的数据集和Dataloader
    transforms = None
    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST,
        cfg.DATASET.DATA_FORMAT,
        cfg.DATASET.NUM_JOINTS,
        cfg.DATASET.GET_RESCORE_DATA,
        transforms,
        bbox_file=cfg.TEST.BBOX_FILE if cfg.TEST.BBOX_GROUPING else None
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
