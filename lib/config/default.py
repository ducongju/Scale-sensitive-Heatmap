# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# Modified by Congju Du (ducongju@hust.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VERBOSE = True
_C.DIST_BACKEND = 'nccl'
_C.MULTIPROCESSING_DISTRIBUTED = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.USE_PRM = False
_C.MODEL.RESOLUTION64 = False

_C.LOSS = CN()
_C.LOSS.NUM_STAGES = 1
_C.LOSS.WITH_HEATMAPS_LOSS = (True,)
_C.LOSS.HEATMAPS_LOSS_FACTOR = (1.0,)

_C.LOSS.WITH_OFFSETS_LOSS = (True,)
_C.LOSS.OFFSETS_LOSS_FACTOR = (1.0,)

_C.LOSS.USE_FOCAL_LOSS = False
_C.LOSS.FOCAL_LOSS_FACTOR = [0.01, 0.1, 0.02]
_C.LOSS.HEATMAP_MIDDLE_LOSS = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'coco_kpt'
_C.DATASET.DATASET_TEST = ''
_C.DATASET.NUM_JOINTS = 17
_C.DATASET.MAX_NUM_PEOPLE = 30
_C.DATASET.TRAIN = 'train2017'
_C.DATASET.TEST = 'val2017'
_C.DATASET.GET_RESCORE_DATA = False
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.USE_MASK = False
_C.DATASET.USE_BBOX_CENTER = False
_C.DATASET.OFFSET_REG = False
_C.DATASET.OFFSET_RADIUS = 4
_C.DATASET.BG_WEIGHT = [1.0]

# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.SCALE_AWARE_SIGMA = False
_C.DATASET.INTER_SIGMA = False
_C.DATASET.BASE_INTERSIZE = 128
_C.DATASET.BASE_INTERSIGMA = 2.0
_C.DATASET.INTER_LINE = False
_C.DATASET.DISTANCE_BASED = False

_C.DATASET.INTRA_SIGMA = False
_C.DATASET.BASE_INTRASIZE = 0.062
_C.DATASET.BASE_INTRASIGMA = 2.0
_C.DATASET.INTRA_CUT = False
_C.DATASET.USE_JNT = False
_C.DATASET.PAUTA = 3
_C.DATASET.JNT_THR = 0.01
_C.DATASET.USE_INT = False
_C.DATASET.SHAPE = False
_C.DATASET.SHAPE_WEIGHT = 1.5

_C.DATASET.USE_SUBSET = False
_C.DATASET.SUBSET_FACTOR = 0.3

_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = [128, 256, 512]
_C.DATASET.FLIP = 0.5

# heatmap generator (default is OUTPUT_SIZE/64)
_C.DATASET.SIGMA = [2.0,]
_C.DATASET.CENTER_SIGMA = 4
_C.DATASET.BASE_SIZE = 256.0
_C.DATASET.BASE_SIGMA = 2.0
_C.DATASET.MIN_SIGMA = 1
_C.DATASET.WITH_CENTER = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.IMAGES_PER_GPU = 32

_C.TEST.FLIP_TEST = True
_C.TEST.SCALE_FACTOR = [1]
# group
_C.TEST.MODEL_FILE = ''
_C.TEST.IGNORE_CENTER = False
_C.TEST.NMS_KERNEL = 3
_C.TEST.NMS_PADDING = 1
_C.TEST.BBOX_GROUPING = False
_C.TEST.BBOX_FILE = ''

# for reg group
_C.TEST.REG_GROUP = True
_C.TEST.USE_HEATMAP = False
_C.TEST.REG_THRESHOLD = 0.98
_C.TEST.DIST_THRESHOLD = 10
_C.TEST.OVERLAP_THRESHOLD = 10
_C.TEST.USE_DECREASE_SCORE = True
_C.TEST.SCALE_DECREASE = 0.001

_C.TEST.KEYPOINT_THRESHOLD = 0.01
_C.TEST.ADJUST_THRESHOLD = 0.05
_C.TEST.MAX_ABSORB_DISTANCE = 75

_C.TEST.POOL_THRESHOLD1 = 300
_C.TEST.POOL_THRESHOLD2 = 200

_C.TEST.GUASSIAN_KERNEL = 6
_C.TEST.GUASSIAN_SIGMA = 2.0

_C.TEST.WITH_HEATMAPS = (True,)

_C.TEST.LOG_PROGRESS = True

_C.RESCORE = CN()
_C.RESCORE.USE = True
_C.RESCORE.END_EPOCH = 20
_C.RESCORE.LR = 0.001
_C.RESCORE.HIDDEN_LAYER = 256
_C.RESCORE.BATCHSIZE = 1024
_C.RESCORE.MODEL_ROOT = 'model/rescore/'
_C.RESCORE.MODEL_FILE = 'model/rescore/final_rescore_coco_kpt.pth'
_C.RESCORE.DATA_FILE = 'data/rescore_data/rescore_dataset_train_coco_kpt'

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = True
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = True
_C.DEBUG.SAVE_HEATMAPS_PRED = True
_C.DEBUG.SAVE_TAGMAPS_PRED = True


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if not os.path.exists(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.join(
            cfg.DATA_DIR, cfg.DATASET.ROOT
        )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    if cfg.DATASET.WITH_CENTER:
        cfg.DATASET.NUM_JOINTS += 1
        cfg.MODEL.NUM_JOINTS = cfg.DATASET.NUM_JOINTS

    if not isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)):
        cfg.DATASET.OUTPUT_SIZE = [cfg.DATASET.OUTPUT_SIZE]

    if not isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)):
        cfg.LOSS.WITH_HEATMAPS_LOSS = (cfg.LOSS.WITH_HEATMAPS_LOSS)

    if not isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.HEATMAPS_LOSS_FACTOR = (cfg.LOSS.HEATMAPS_LOSS_FACTOR)
    
    if not isinstance(cfg.LOSS.WITH_OFFSETS_LOSS, (list, tuple)):
        cfg.LOSS.WITH_OFFSETS_LOSS = (cfg.LOSS.WITH_OFFSETS_LOSS)
    if not isinstance(cfg.LOSS.OFFSETS_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.OFFSETS_LOSS_FACTOR = (cfg.LOSS.OFFSETS_LOSS_FACTOR)

    cfg.freeze()


def check_config(cfg):
    assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.WITH_HEATMAPS_LOSS), \
        'LOSS.NUM_SCALE should be the same as the length of LOSS.WITH_HEATMAPS_LOSS'
    assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.HEATMAPS_LOSS_FACTOR), \
        'LOSS.NUM_SCALE should be the same as the length of LOSS.HEATMAPS_LOSS_FACTOR'
    assert cfg.LOSS.NUM_STAGES == len(cfg.TEST.WITH_HEATMAPS), \
        'LOSS.NUM_SCALE should be the same as the length of TEST.WITH_HEATMAPS'
    assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.OFFSETS_LOSS_FACTOR), \
        'LOSS.NUM_SCALE should be the same as the length of LOSS.OFFSETS_LOSS_FACTOR'


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
