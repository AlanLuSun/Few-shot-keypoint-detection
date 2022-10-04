# -*- coding: utf-8 -*-
import torch
from torch import nn
from torchvision.models.resnet import resnet34, resnet50
from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg16
from network.functions import ReverseLayerF
from torchvision.models.resnet import Bottleneck

import torch.utils.model_zoo as model_zoo

from utils import print_weights, make_grid_images, make_uncertainty_map, save_plot_image

import math
import time

from heatmap import putGaussianMaps
import copy
import numpy as np


from network.hrnet_config import cfg
from network.hrnet_config import update_config
import argparse

from network.pose_hrnet import PoseHighResolutionNet

class PyramidPooling(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PyramidPooling, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                # nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(torch.nn.functional.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class EncoderHRNet(nn.Module):
    def __init__(self, hrnet_cfg, trunk='hrnet-w48', layer_to_freezing=-1, downsize_factor=16, use_ppm=False, ppm_remain_dim=True, specify_output_channel=None, lateral_output_mode_layer=(0, 6)):
        super(EncoderHRNet, self).__init__()


        if trunk == 'hrnet-w48':



            self.hrnet_backbone = PoseHighResolutionNet(hrnet_cfg, layer_to_freezing, lateral_output_mode_layer[1])
            self.hrnet_backbone.init_weights(pretrained=hrnet_cfg.MODEL.PRETRAINED)

            feature_dim = 384
        elif trunk == 'hrnet-w32':
            self.hrnet_backbone = PoseHighResolutionNet(hrnet_cfg, layer_to_freezing, lateral_output_mode_layer[1])
            self.hrnet_backbone.init_weights(pretrained=hrnet_cfg.MODEL.PRETRAINED)

            feature_dim = 256

        # self.encoder_list2 = nn.Sequential(
        #     resnet.layer3._modules['0'],
        #     resnet.layer3._modules['1'],  # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
        # )
        self.freeze = layer_to_freezing
        self.use_ppm = use_ppm
        self.ppm_remain_dim = ppm_remain_dim
        if use_ppm:
            # feature_dim = 1024
            if self.ppm_remain_dim == True:
                feature_dim_half = int(feature_dim / 2)  # 512
                self.conv1x1 = nn.Conv2d(feature_dim, feature_dim_half, kernel_size=1, stride=1,
                                         padding=0)  # output feature dim will be 512
                bins = [1, 4, 8, 12]
                self.ppm = PyramidPooling(feature_dim_half, int(feature_dim_half / len(bins)),
                                          bins)  # note output feature dim will be 512 * 2 = 1024
            else:
                bins = [1, 4, 8, 12]
                self.ppm = PyramidPooling(feature_dim, int(feature_dim / len(bins)),
                                          bins)  # note output feature dim will be 1024 * 2 = 2048

        self.specify_output_channel = specify_output_channel
        if specify_output_channel != None:
            self.end_conv1x1 = nn.Sequential(
                nn.Conv2d(feature_dim, specify_output_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(specify_output_channel),
                nn.ReLU(True)
            )

        self.lateral_output_mode = lateral_output_mode_layer[0]  # if 0, don't use lateral output
        self.lateral_output_layer = lateral_output_mode_layer[1]
        lateral_channel_in = feature_dim # 384 for hrnet-w48; 256 for hrnet-w32
        lateral_stride = 1 # layer <= 6, c=1024, h=24x24; lateral_stride can downsample the feature into 12 x 12
        if self.lateral_output_mode == 1:  # only using image
            # semantic distinctiveness module
            self.sdm = nn.Sequential(
                nn.Conv2d(lateral_channel_in, 1024, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(1024),
                nn.ReLU(True),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=lateral_stride, padding=1),
                # nn.BatchNorm2d(1024),
                nn.ReLU(True),
                nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
            )
            for m in self.sdm.modules():
                if isinstance(m, nn.Conv2d):
                    # m.weight.data.normal_(mean=0, std=0.01)
                    nn.init.kaiming_normal_(m.weight)
                    # if m.bias is not None:
                    #     # m.bias.data.zero_()
                    #     nn.init.kaiming_normal_(m.bias)


        print('Encoder initialized!')
        # print(self.encoder_list)


    def forward(self, x: torch.Tensor, x2=None, enable_lateral_output=False):
        """feedforward propagation

        Arguments:
            x {torch.Tensor} -- input samples batch × 3 × height × width

        Returns:
            [torch.Tensor] --
        """
        # for i in range(len(self.encoder_list)):
        #     x = self.encoder_list[i](x)

        x, x_lateral = self.hrnet_backbone(x)

        # print_weights(self.encoder_list[0].weight.data)
        # out = self.encoder_list(x)
        # if self.freeze:
        #     out = out.detach()
        # out = self.encoder_list2(out)
        # print_weights(self.encoder_list2[0].conv1.weight.data)

        if self.use_ppm:
            if self.ppm_remain_dim:
                x = self.conv1x1(x)  # half the channel
            x = self.ppm(x)

        if self.specify_output_channel != None:
            x = self.end_conv1x1(x)

        if enable_lateral_output == True:
            if self.lateral_output_mode == 1:    # only using image
                lateral_output = self.sdm(x_lateral)
            else:
                lateral_output = None
        else:
            lateral_output = None

        # out = self.convs(out)
        # out = self.fc(out.reshape(-1, 256 * 4 * 4))

        # out = self.encoder_list(x)
        #
        # identity = out
        #
        # out = self.conv1(out)
        # out = self.bn1(out)
        # out = self.relu1(out)
        # out = self.maxpool1(out)
        #
        # out = self.conv2(out)
        # out = self.bn2(out)
        #
        # identity = self.downsample(identity)
        # out += identity
        #
        # out = self.relu2(out)
        # out = self.maxpool2(out)
        #
        # out = self.fc(out.reshape(-1, 256 * 5 * 5))



        return x, lateral_output

