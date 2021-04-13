# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcn import DeformConv, ModulatedDeformConv

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class STNBLOCK(nn.Module):
    """
    pixel-wise STN是 Deformable Convoultion的一个套壳，先是经过几个卷积模块预测
    卷积核内各元素偏移后的位置，然后由此计算得到偏移量，输入到Deformable Conv中计算
    """
    expansion = 1
    def __init__(self, inplanes, outplanes, stride=1, 
            downsample=None, dilation=1, deformable_groups=1):
        super(STNBLOCK, self).__init__()
        
        # 3*3的卷积核内位置相对中心位置的偏移可编码为2*9的矩阵，维度2为x和y上偏移，维度9为元素数目
        regular_matrix = torch.tensor(np.array([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
                                                [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]]))
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample

        # 转换矩阵T的维度为2*2,所以在这里将channel设为 2*2=4 
        self.transform_matrix_conv1 = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.stn_conv1 = DeformConv(
            inplanes,
            outplanes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.bn1 = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
 
        self.transform_matrix_conv2 = nn.Conv2d(outplanes, 4, 3, 1, 1, bias=True)            
        self.stn_conv2 = DeformConv(
            outplanes,
            outplanes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups)
        self.bn2 = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
 
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        residual = x
        (N,C,H,W) = x.shape
        transform_matrix1 = self.transform_matrix_conv1(x)
        transform_matrix1 = transform_matrix1.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset1 = torch.matmul(transform_matrix1, self.regular_matrix)
        offset1 = offset1-self.regular_matrix
        offset1 = offset1.transpose(1,2)
        offset1 = offset1.reshape((N,H,W,18)).permute(0,3,1,2)
 
        out = self.stn_conv1(x, offset1)
        out = self.bn1(out)
        out = self.relu(out)
 
        transform_matrix2 = self.transform_matrix_conv2(x)
        transform_matrix2 = transform_matrix2.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset2 = torch.matmul(transform_matrix2, self.regular_matrix)
        offset2 = offset2-self.regular_matrix
        offset2 = offset2.transpose(1,2)
        offset2 = offset2.reshape((N,H,W,18)).permute(0,3,1,2)
        out = self.stn_conv2(out, offset2)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

        
class HighResolutionModule(nn.Module):
    """
    模块作用是按照配置要求构建HRNet部分,建立相应的分支数并完成各分支之间的特征融合
    num_branches是分支数目,blocks是基础卷积模块,num_blocks是每个分支上模块数目,
    num_inchannels是每个分支上每个模块的输入通道数目
    multi_scale_output控制是否在输出的时侯结合多个分支的结果
    fuse_method控制多分支融合的方法，默认为相加ADD
    """
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        # 控制通道数一致来完成残差模块的组成
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        # 第一个block的输入可能和其他不同，故单独构建
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion

        """
        若expansion = 1,则使用Basic Block,此时输出channel为num_channels
        若expansion != 1,则使用Bottleneck Block,此时输出channel为num_channels * expansion
        """
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        """
        如果需要多分支的结果结合输出，则需要对每个分支设置一个fuse_layer
        分支序号越小，则分辨率相对越高
        """
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                # 如果分辨率比指定分支分辨率低，则需要上采样操作
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                # 如果分辨率和指定分支分辨率相同，则不作任何操作
                elif j == i:
                    fuse_layer.append(None)
                # 如果分辨率比指定分支分辨率高，则需要下采样操作
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        # 只有在最后一层完成分支间的通道转换，否则只在同分支内做下采样
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        # 把所有branch的结果融合(ADD)
        for i in range(len(self.fuse_layers)):
            # 由于fuse_layer[0][0]为None,所以作为特殊情况提出
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                # 同理，避免None的情况
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse
