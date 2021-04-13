# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Congju Du and Han Yu (ducongju@hust.edu.cn; yuhan2019@hust.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import csv
import os
import shutil
import time
import sys
sys.path.append("../lib")

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models
import math

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapRegParser
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


CROWDPOSE_KEYPOINT_INDEXES = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
    6: 'left_hip',
    7: 'right_hip',
    8: 'left_knee',
    9: 'right_knee',
    10: 'left_ankle',
    11: 'right_ankle',
    12: 'head',
    13: 'neck'
}


COCO_PERSON_SKELETON = [
    (0, 1), (1, 3), (3, 5),         # left head
    (0, 2), (2, 4), (4, 6),         # right head
    (0, 5), (5, 7), (7, 9),         # left arm
    (0, 6), (6, 8), (8, 10),        # right arm
    (5, 6),                         # l shoulder to r shoulder
    (12, 11),                       # r hip to l hip
    (5, 11), (11, 13), (13, 15),    # left side
    (6, 12), (12, 14), (14, 16),    # rught side
]


def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    parser = HeatmapRegParser(cfg)

    with torch.no_grad():
        heatmap_fuse = 0
        final_heatmaps = None
        final_kpts = None
        input_size = cfg.DATASET.INPUT_SIZE

        for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
            #joints, mask do not use in demo mode
            joints = np.zeros((0, cfg.DATASET.NUM_JOINTS, 3))
            mask = np.zeros((image.shape[0], image.shape[1]))
            image_resized, _, _, center, scale = resize_align_multi_scale(
                image, joints, mask, input_size, s, 1.0
            )
            image_resized_copy = image_resized

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            outputs, heatmaps, kpts = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            final_heatmaps, final_kpts = aggregate_results(
                cfg, final_heatmaps, final_kpts, heatmaps, kpts
            )

        for heatmap in final_heatmaps:
            heatmap_fuse += up_interpolate(
                heatmap,
                size=(base_size[1], base_size[0]),
                mode='bilinear'
            )
        heatmap_fuse = heatmap_fuse/float(len(final_heatmaps))

        # for only pred kpts
        grouped, scores = parser.parse(
            final_heatmaps, final_kpts, heatmap_fuse[0], use_heatmap=False
        )
        if len(scores) == 0:
            return []

        results = get_final_preds(
            grouped, center, scale,
            [heatmap_fuse.size(-1), heatmap_fuse.size(-2)]
        )

        final_results = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(results[i])

        if len(final_results) == 0:
            return []
    return final_results, heatmap_fuse, image_resized_copy


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    # if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
    #     shutil.rmtree(pose_dir)
    # os.makedirs(pose_dir, exist_ok=True)
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)
    return pose_dir


def plot_pose(img, person_list, bool_fast_plot=True):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
              [255, 0, 85], [255, 0, 170], [255, 0, 255],
              [0, 255, 0], [85, 255, 0], [170, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255],
              [0, 0, 255], [0, 85, 255], [0, 170, 255],
              [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 255, 255]]

    image = img.copy()
    limb_thickness = 2

    for limb_type in range(len(COCO_PERSON_SKELETON)):
        # hide the limb from nose to arm
        if limb_type == 6 or limb_type == 9:
            continue
        for person_joint_info in person_list:
            joint1 = person_joint_info[COCO_PERSON_SKELETON[limb_type][0]].astype(int)
            joint2 = person_joint_info[COCO_PERSON_SKELETON[limb_type][1]].astype(int)
            if joint1[-1] == -1 or joint2[-1] == -1:
                continue
            
            joint_coords = [joint1[:2], joint2[:2]]
            for joint in joint_coords:
                cv2.circle(image, tuple(joint.astype(
                    int)), 4, (255, 255, 255), thickness=-1)

            # mean along the axis=0 computes mean Y coord and mean X coord
            coords_center = tuple(
                np.round(np.mean(joint_coords, 0)).astype(int))

            limb_dir = joint_coords[0] - joint_coords[1]
            limb_length = np.linalg.norm(limb_dir)
            # Get the angle of limb_dir in degrees using atan2
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))

            cur_image = image if bool_fast_plot else image.copy()
            polygon = cv2.ellipse2Poly(
                coords_center, (int(limb_length / 2), limb_thickness),
                int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_image, polygon, colors[limb_type])

            if not bool_fast_plot:
                image = cv2.addWeighted(image, 0.4, cur_image, 0.6, 0)

    # to_plot is the location of all joints found overlaid of image
    if bool_fast_plot:
        to_plot = image.copy()
    else:
        to_plot = cv2.addWeighted(img, 0.3, image, 0.7, 0)
    return to_plot, image

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--imageFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--visthre', type=float, default=0.3)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # Loading an image
    # image
    image = cv2.imread(args.imageFile)
    print("image:", image.shape)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("image_rgb:", image_rgb.shape)
    image_pose = image_rgb.copy()
    image_plot = image.copy()

    now = time.time()
    pose_preds, heatmap_fuse, image_resized_copy = get_pose_estimation_prediction(
        cfg, pose_model, image_pose, args.visthre, transforms=pose_transform)
    then = time.time()

    for coords in pose_preds:
        # Draw each point on image
        for coord in coords:
            x_coord, y_coord = int(coord[0]), int(coord[1])
            cv2.circle(image_plot, (x_coord, y_coord), 4, (255, 0, 0), 2)
    image_plot, _ = plot_pose(image_plot, pose_preds)
    
    img_file = os.path.join(pose_dir, 'pose_{}_{}.jpg'.format(str(args.imageFile.split('.')[-2].split('/')[-1]), int(time.time())))
    cv2.imwrite(img_file, image_plot)

    # heatmap
    gray_img = np.array(heatmap_fuse[0][cfg.DATASET.NUM_JOINTS - 1].cpu())
    for i in range(cfg.DATASET.NUM_JOINTS - 1):
        gray_img += np.array(heatmap_fuse[0][i].cpu())
    gray_img = np.clip(gray_img * 255, 0, 255).astype(np.uint8)

    heat_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    heatmap_add = cv2.addWeighted(image_resized_copy, 0.3, heat_img, 0.7, 0)

    img_file2 = os.path.join(pose_dir, 'heatmap_{}_{}.jpg'.format(str(args.imageFile.split('.')[-2].split('/')[-1]), int(time.time())))
    cv2.imwrite(img_file2, heatmap_add)

    # groundtruth
    

    dataset_dir = "./data/coco"
    coco = COCO(os.path.join(dataset_dir,'annotations','person_keypoints_val2017.json'))
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = int(args.imageFile.split('.')[-2].split('/')[-1])
    img = coco.loadImgs(imgIds)[0]
    image_path = args.imageFile
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    
    for i in range(len(anns)):
        pose_gt = np.array(anns[i]['keypoints']).reshape(1, 17, 3)
        image_plot2, _ = plot_pose(image_plot2, pose_gt)
    img_file3 = os.path.join(pose_dir, 'groundtruth_{}_{}.jpg'.format(str(args.imageFile.split('.')[-2].split('/')[-1]), int(time.time())))
    cv2.imwrite(img_file3, image_plot2)


if __name__ == '__main__':
    main()
