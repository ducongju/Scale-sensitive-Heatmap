# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# Modified by Congju Du (ducongju@hust.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from .CrowdPoseDataset import CrowdPoseDataset
from .target_generators import HeatmapGenerator


logger = logging.getLogger(__name__)


class CrowdPoseKeypoints(CrowdPoseDataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 remove_images_without_annotations,
                 heatmap_generator,
                 offset_generator=None,
                 transforms=None):
        super().__init__(cfg.DATASET.ROOT,
                         dataset_name,
                         cfg.DATASET.DATA_FORMAT,
                         cfg.DATASET.NUM_JOINTS,
                         cfg.DATASET.GET_RESCORE_DATA)

        if cfg.DATASET.WITH_CENTER:
            assert cfg.DATASET.NUM_JOINTS == 15, 'Number of joint with center for CrowdPose is 15'
        else:
            assert cfg.DATASET.NUM_JOINTS == 14, 'Number of joint for CrowdPose is 14'

        self.num_scales = self._init_check(heatmap_generator)
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.with_center = cfg.DATASET.WITH_CENTER
        self.num_joints_without_center = self.num_joints - 1 \
            if self.with_center else self.num_joints
        self.base_sigma = cfg.DATASET.BASE_SIGMA
        self.base_size = cfg.DATASET.BASE_SIZE
        self.min_sigma = cfg.DATASET.MIN_SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.sigma = cfg.DATASET.SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.use_mask = cfg.DATASET.USE_MASK
        self.use_bbox_center = cfg.DATASET.USE_BBOX_CENTER

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator
        #######################################################################
        self.scale_aware_sigma = cfg.DATASET.SCALE_AWARE_SIGMA
        self.inter_sigma = cfg.DATASET.INTER_SIGMA
        self.base_intersize = cfg.DATASET.BASE_INTERSIZE
        self.base_intersigma = cfg.DATASET.BASE_INTERSIGMA
        self.inter_line = cfg.DATASET.INTER_LINE
        self.distance_based = cfg.DATASET.DISTANCE_BASED

        self.intra_sigma = cfg.DATASET.INTRA_SIGMA
        self.base_intrasize = cfg.DATASET.BASE_INTRASIZE
        self.base_intrasigma = cfg.DATASET.BASE_INTRASIGMA
        self.intra_cut = cfg.DATASET.INTRA_CUT
        #######################################################################

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)

        mask = self.get_mask(anno, idx)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        joints, area = self.get_joints(anno)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()
        ind_mask_list = list()
        offset_list = list()
        weights_list = list()

        if self.transforms:
            img, mask_list, joints_list, area = self.transforms(
                img, mask_list, joints_list, area
            )

        for scale_id in range(self.num_scales):
            scaled_target = []
            scaled_mask = []
            mask = mask_list[scale_id].copy()

            for i, sgm in enumerate(self.sigma[scale_id]):
                target_t, ignored_t = self.heatmap_generator[scale_id](
                    joints_list[scale_id],
                    sgm,
                    self.center_sigma,
                    self.bg_weight[scale_id][i])

                scaled_mask.append((mask*ignored_t).astype(np.float32))
                scaled_target.append(target_t.astype(np.float32))

            if self.offset_generator is not None:
                offset_t, weight_t = self.offset_generator[scale_id](
                    joints_list[scale_id], area)
                offset_list.append([offset_t])
                weights_list.append([weight_t])

            target_list.append(scaled_target)
            ind_mask_list.append(scaled_mask)

        return img, target_list, ind_mask_list, offset_list, weights_list

    def get_joints(self, anno):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        # joints = np.zeros((num_people, self.num_joints, 3))
        ###########################################################################
        if self.scale_aware_sigma:
            joints = np.zeros((num_people, self.num_joints, 4))
        else:
            joints = np.zeros((num_people, self.num_joints, 3))
        ###########################################################################

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints_without_center, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

            if self.use_mask == True:
                area[i, 0] = obj['area']
            else:
                area[i, 0] = obj['bbox'][2]*obj['bbox'][3]

            if self.with_center:
                bbox = obj['bbox']
                center_x = (2*bbox[0] + bbox[2]) / 2.
                center_y = (2*bbox[1] + bbox[3]) / 2.
                joints_sum = np.sum(joints[i, :-1, :2], axis=0)
                num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
                if self.use_bbox_center or num_vis_joints <= 0:
                    joints[i, -1, 0] = center_x
                    joints[i, -1, 1] = center_y
                else:
                    joints[i, -1, :2] = joints_sum / num_vis_joints
                joints[i, -1, 2] = 1

            ###########################  人体外部尺度  ################################
            if self.scale_aware_sigma:

                # 人体外部尺度
                intersigma = np.ones(num_people)
                if self.inter_sigma:
                    intersize = np.ones(num_people)
                    # 基于距离
                    if self.distance_based and joints[i, 0, 0] != 0 and joints[i, 0, 1] != 0 and joints[i, 1, 0] != 0 and joints[i, 1, 1] != 0 and joints[i, 10, 0] != 0 and joints[i, 10, 1] != 0 and joints[i, 11, 0] != 0 and joints[i, 11, 1] != 0:
                        centerx_eye = (joints[i, 0, 0] + joints[i, 1, 0]) / 2
                        centery_eye = (joints[i, 0, 1] + joints[i, 1, 1]) / 2
                        p1x = joints[i, 1, 0] - joints[i, 0, 0]
                        p1y = joints[i, 1, 1] - joints[i, 0, 1]
                        p2x = p1y
                        p2y = -p1x
                        p3x = joints[i, 10, 0] - centerx_eye
                        p3y = joints[i, 10, 1] - centery_eye
                        p4x = joints[i, 11, 0] - centerx_eye
                        p4y = joints[i, 11, 1] - centery_eye
                        import math
                        if (p2x ** 2 + p2y ** 2) == 0:
                            intersize[i] = obj['bbox'][2]
                        else:
                            d1 = (p3x * p2x + p3y * p2y) / math.sqrt(p2x ** 2 + p2y ** 2)
                            d2 = (p4x * p2x + p4y * p2y) / math.sqrt(p2x ** 2 + p2y ** 2)
                            intersize[i] = max(abs(d1), abs(d2)).astype('int')
                        # print(intersize[i])
                        # if intersize[i] <= 0:
                        #    intersize[i] = obj['bbox'][2]
                    # 基于面积
                    else:
                        # intersize[i] = area[i, 0]
                        # intersize[i] = obj['bbox'][2]*obj['bbox'][3]
                        intersize[i] = obj['bbox'][2]
                        # intersize[i] = self.base_intersize
                    # 截断设置
                    if self.base_intersize > intersize[i]:
                        intersize[i] = self.base_intersize
                    # 线性变化
                    if self.inter_line:
                        intersigma = intersize / self.base_intersize * self.base_intersigma
                    # 非线性变化
                    else:
                        x = intersize / self.base_intersize
                        intersigma = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) * self.base_intersigma

                # 人体内部尺度
                intrasigma = np.ones(14)
                if self.intra_sigma:
                    # intrasize = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
                    intrasize = np.array([.079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089, .050, .050])
                    # 截断设置
                    if self.intra_cut:
                        for j in range(14):
                            if self.base_intrasize > intrasize[j]:
                                intrasize[j] = self.base_intrasize             
                    intrasigma = intrasize / self.base_intrasize * self.base_intrasigma

                # 人体综合尺度
                for j in range(14):
                    joints[i, j, 3] = intersigma[i] * intrasigma[j]
            ###########################  人体内部尺度  ################################

        return joints, area

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']))

        return m < 0.5

    def _init_check(self, heatmap_generator):
        assert isinstance(heatmap_generator, (list, tuple)
                          ), 'heatmap_generator should be a list or tuple'
        return len(heatmap_generator)
