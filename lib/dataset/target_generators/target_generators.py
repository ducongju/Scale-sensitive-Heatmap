# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Congju Du (ducongju@hust.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class HeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_heat_val(self, sigma, x, y, x0, y0):

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return g

    def __call__(self, joints, sgm, ct_sgm, bg_weight=1.0):
        '''
        sgm为关节点的标准差；ct_sgm为人体中心点的标准差
        bg_weight为背景元素所占的权重
        '''
        assert self.num_joints == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints

        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        ignored_hms = 2*np.ones((1, self.output_res, self.output_res),
                                dtype=np.float32)

        hms_list = [hms, ignored_hms]

        for p in joints:
            for idx, pt in enumerate(p):
                if idx < self.num_joints:
                    sigma = sgm
                else:
                    sigma = ct_sgm
                if pt[2] > 0:           # 值为0时表明图中关节点不可见
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue
                    # 对坐标位置进行一个筛选

                    # 通过sigma计算一个辐射边界，边界内才有值，边界外认为没有影响
                    ul = int(np.floor(x - 3 * sigma - 1)
                             ), int(np.floor(y - 3 * sigma - 1))
                    br = int(np.ceil(x + 3 * sigma + 2)
                             ), int(np.ceil(y + 3 * sigma + 2))
                    # 半径为1
                    # ul = int(np.floor(x)
                    #          ), int(np.floor(y))
                    # br = int(np.ceil(x+1)
                    #          ), int(np.ceil(y+1))

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    # 半径不限
                    # cc, dd = 0, self.output_res
                    # aa, bb = 0, self.output_res

                    joint_rg = np.zeros((bb-aa, dd-cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy-aa, sx -
                                     cc] = self.get_heat_val(sigma, sx, sy, x, y)

                    # 关节热图的边界框内的值不能会负数, 且重叠位置使用最大值
                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(
                        hms_list[0][idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][0, aa:bb, cc:dd] = 1.

        # 通过值的差别来得到没有关节影响辐射区域的位置
        hms_list[1][hms_list[1] == 2] = bg_weight

        return hms_list


###########################################################################################
class ScaleAwareHeatmapGenerator():
    def __init__(self, output_res, num_joints, use_jnt, jnt_thr, use_int, shape, shape_weight, pauta):
        self.output_res = output_res
        self.num_joints = num_joints
        self.use_jnt = use_jnt
        self.jnt_thr = jnt_thr
        self.use_int = use_int
        self.shape = shape
        self.shape_weight = shape_weight
        self.pauta = pauta

    def get_heat_val(self, sigma, x, y, x0, y0):

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return g

    def __call__(self, joints, sgm, ct_sgm, bg_weight=1.0):
        '''
        sgm为关节点的标准差；ct_sgm为人体中心点的标准差
        bg_weight为背景元素所占的权重
        '''
        assert self.num_joints == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints

        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        ignored_hms = 2*np.ones((1, self.output_res, self.output_res),
                                dtype=np.float32)

        hms_list = [hms, ignored_hms]

        for p in joints:
            # print("len(p)", len(p))
            # print("self.num_joints", self.num_joints)
            for idx, pt in enumerate(p):                
                if idx < self.num_joints - 1:
                    #####################################################
                    sigma = pt[3]
                    #####################################################
                else:
                    sigma = ct_sgm

                if pt[2] > 0:           # 值为0时表明图中关节点不可见
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue
                    # 对坐标位置进行一个筛选
                    if self.use_jnt:
                        ################### JNT ######################
                        radius = np.sqrt(np.log(1 / self.jnt_thr) * 2 * sigma ** 2)
                        if self.use_int:
                            radius = int(np.floor(radius))  # 取整
                        
                        if self.shape:
                            if idx == (0,3,4):
                                ul = int(np.floor(x - radius - 1)), int(np.floor(y - self.shape_weight * radius - 1))
                                br = int(np.ceil(x + radius + 2)), int(np.ceil(y + self.shape_weight * radius + 2))
                                # print(idx, ul, br)
                            elif idx == (1,2):
                                ul = int(np.floor(x - self.shape_weight * radius - 1)), int(np.floor(y - radius - 1))
                                br = int(np.ceil(x + self.shape_weight * radius + 2)), int(np.ceil(y + radius + 2))
                            else:
                                ul = int(np.floor(x - radius - 1)), int(np.floor(y - radius - 1))
                                br = int(np.ceil(x + radius + 2)), int(np.ceil(y + radius + 2))
                        
                        else:
                            ul = int(np.floor(x - radius - 1)), int(np.floor(y - radius - 1))
                            br = int(np.ceil(x + radius + 2)), int(np.ceil(y + radius + 2))
                        ################### JNT ######################

                    else:
                        ################### 3sigma ######################
                        # 通过sigma计算一个辐射边界，边界内才有值，边界外s认为没有影响
                        ul = int(np.floor(x - self.pauta * sigma - 1)), int(np.floor(y - self.pauta * sigma - 1))
                        br = int(np.ceil(x + self.pauta * sigma + 2)), int(np.ceil(y + self.pauta * sigma + 2))
                        ################### 3sigma ######################

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                    joint_rg = np.zeros((bb-aa, dd-cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy - aa, sx - cc] = self.get_heat_val(sigma, sx, sy, x, y)

                    # 关节热图的边界框内的值不能会负数, 且重叠位置使用最大值
                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(hms_list[0][idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][0, aa:bb, cc:dd] = 1.

        # 通过值的差别来得到没有关节影响辐射区域的位置
        hms_list[1][hms_list[1] == 2] = bg_weight

        return hms_list
##########################################################################################


class OffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius):
        self.num_joints_without_center = num_joints - 1
        self.output_w = output_w
        self.output_h = output_h
        self.num_joints = num_joints
        self.radius = radius

    def __call__(self, joints, area):
        assert joints.shape[1] == self.num_joints, \
            'the number of joints should be 18, 17 keypoints + 1 center joint.'

        offset_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)

        for person_id, p in enumerate(joints):
            # p[-1, 0]为人体中心点
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue

            for idx, pt in enumerate(p[:-1]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_w or y >= self.output_h:
                        continue
                    
                    # 计算影响区域，用矩阵框来代替圆形区域
                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y

                            if offset_map[idx*2, pos_y, pos_x] != 0 \
                                    or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area[person_id]:
                                    continue
                            offset_map[idx*2, pos_y, pos_x] = offset_x
                            offset_map[idx*2+1, pos_y, pos_x] = offset_y
                            
                            weight_map[idx*2, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            weight_map[idx*2+1, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            area_map[pos_y, pos_x] = area[person_id]

        return offset_map, weight_map
