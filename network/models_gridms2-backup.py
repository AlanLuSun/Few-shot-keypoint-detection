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

class Encoder(nn.Module):
    def __init__(self, trunk='resnet50', layer_to_freezing=-1, downsize_factor=16, use_ppm=False, ppm_remain_dim=True, specify_output_channel=None, lateral_output_mode_layer=(0, 6)):
        super(Encoder, self).__init__()

        if trunk == 'resnet50':
            # Bottleneck numbers [3, 4, 6, 3] for layer 1, 2, 3, 4
            resnet = resnet50(pretrained=True)

            if downsize_factor == 16:
                self.encoder_list = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3._modules['0'],
                    resnet.layer3._modules['1'],  # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
                    # resnet.layer3,
                    # resnet.layer4._modules['0'],
                    # resnet.layer4._modules['1'],    # feature size B x 2048 x 12 x 12, 1/32 (368 x 368)
                )
                feature_dim = 1024  # B x 1024 x 23 x 23 (1/16)
            elif downsize_factor == 32:
                self.encoder_list = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    # resnet.layer3._modules['0'],
                    # resnet.layer3._modules['1'],  # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
                    resnet.layer3,
                    resnet.layer4._modules['0'],
                    resnet.layer4._modules['1'],    # feature size B x 2048 x 12 x 12, 1/32 (368 x 368)
                    # resnet.layer4
                )
                feature_dim = 2048  # B x 2048 x 12 x 12 (1/32)
            elif downsize_factor == 46:
                self.encoder_list = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3,
                    resnet.layer4._modules['0'],
                    resnet.layer4._modules['1'],
                    nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(2048),
                    nn.ReLU(True),
                    nn.Conv2d(2048, 3072, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(3072),
                    nn.ReLU(True),
                )
                feature_dim = 3072  # B x 3072 x 8 x 8 (1/46)
                self.encoder_list[9].weight.data.normal_(mean=0, std=0.01)
                self.encoder_list[12].weight.data.normal_(mean=0, std=0.01)

            elif downsize_factor == 64:
                self.encoder_list = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3,
                    resnet.layer4._modules['0'],
                    resnet.layer4._modules['1'],
                    # nn.Conv2d(2048, 4096, kernel_size=1, stride=1, padding=0),
                    # nn.BatchNorm2d(4096),
                    # nn.ReLU(True),
                    # nn.AdaptiveAvgPool2d((6, 6))  # B x 2048 x 6 x 6 (1/64),
                    nn.Conv2d(2048, 4096, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(4096),
                    nn.ReLU(True),
                )
                feature_dim = 4096  # B x 4096 x 6 x 6 (1/64)
                self.encoder_list[9].weight.data.normal_(mean=0, std=0.01)

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
                    self.conv1x1 = nn.Conv2d(feature_dim, feature_dim_half, kernel_size=1, stride=1, padding=0)  # output feature dim will be 512
                    bins = [1, 4, 8, 12]
                    self.ppm = PyramidPooling(feature_dim_half, int(feature_dim_half/len(bins)), bins) # note output feature dim will be 512 * 2 = 1024
                else:
                    bins = [1, 4, 8, 12]
                    self.ppm = PyramidPooling(feature_dim, int(feature_dim/len(bins)), bins) # note output feature dim will be 1024 * 2 = 2048

            self.specify_output_channel = specify_output_channel
            if specify_output_channel != None:
                self.end_conv1x1 = nn.Sequential(
                    nn.Conv2d(feature_dim, specify_output_channel, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(specify_output_channel),
                    nn.ReLU(True)
                )

            self.lateral_output_mode = lateral_output_mode_layer[0]  # if 0, don't use lateral output
            self.lateral_output_layer = lateral_output_mode_layer[1]
            lateral_channel_in = 1024 if self.lateral_output_layer <= 6 else 2048
            lateral_stride = 2 if self.lateral_output_layer <= 6 else 1  # layer <= 6, c=1024, h=24x24; lateral_stride can downsample the feature into 12 x 12
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
            elif self.lateral_output_mode == 2:  # using image and saliency map
                sdm_extractor_feature_dim = 512
                self.sdm_extractor = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(256, sdm_extractor_feature_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(sdm_extractor_feature_dim),
                    # nn.ReLU(True)
                )
                self.sdm_transition = nn.Sequential(
                    nn.Conv2d(lateral_channel_in, lateral_channel_in, kernel_size=3, stride=lateral_stride, padding=1),
                    nn.BatchNorm2d(lateral_channel_in),
                )
                self.sdm = nn.Sequential(
                    nn.ReLU(True),
                    nn.Conv2d(lateral_channel_in+sdm_extractor_feature_dim, 1024, kernel_size=1, stride=1, padding=0),
                    # nn.BatchNorm2d(1024),
                    nn.ReLU(True),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(1024),
                    nn.ReLU(True),
                    nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
                )
                for m in self.sdm_extractor.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight)
                        # if m.bias is not None:
                        #     m.bias.data.zero_()
                for m in self.sdm.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight)
                        # if m.bias is not None:
                        #     m.bias.data.zero_()
                nn.init.kaiming_normal_(self.sdm_transition[0].weight)
                # nn.init.zeros_(self.sdm_transition[0].bias)


            # # Architecture 1
            # self.convs = nn.Sequential(
            #     nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(512),
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=3, stride=2),
            #     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=3, stride=2)
            #     # nn.AdaptiveAvgPool2d(output_size=(6, 6))
            # )
            #
            # self.fc = nn.Sequential(
            #     # nn.Linear( 256 * 4 * 4, 1024),  # need to adjust mannualy
            #     # nn.BatchNorm1d(1024),
            #     # nn.ReLU(inplace=True),
            #     # nn.Linear(1024, 512),
            #     # nn.BatchNorm1d(512),
            #     # nn.ReLU(inplace=True),
            #     nn.Linear(256 * 4 * 4, 4),
            #     nn.LogSoftmax(dim=1)
            # )

            # Architecture 2
            # self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
            # self.bn1 = nn.BatchNorm2d(512)
            # self.relu1 = nn.ReLU(inplace=True)
            # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            #
            # self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            # self.bn2 = nn.BatchNorm2d(256)
            # self.relu2 = nn.ReLU(inplace=True)
            # self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            #
            # self.downsample = nn.Sequential(
            #     nn.Conv2d(1024, 256, kernel_size=1, stride=2, padding=0),
            #     nn.BatchNorm2d(256)
            # )
            #
            # self.fc = nn.Sequential(
            #     nn.Linear(256 * 5 * 5, 4),
            #     nn.LogSoftmax(dim=1)
            # )



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

        for i in range(len(self.encoder_list)):
            x = self.encoder_list[i](x)
            if self.freeze == i:
                x = x.detach()
            if self.lateral_output_mode > 0 and self.lateral_output_layer == i:
                x_lateral = x  # copy feature here for lateral output

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
            elif self.lateral_output_mode == 2:  # using image and saliency map
                x2 = self.sdm_extractor(x2)
                x_lateral = self.sdm_transition(x_lateral)
                x_lateral_fusion = torch.cat([x_lateral, x2], dim=1)
                lateral_output = self.sdm(x_lateral_fusion)
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

class EncoderMultiScaleType1(nn.Module):
    def __init__(self, trunk='resnet50'):
        super(EncoderMultiScaleType1, self).__init__()

        if trunk == 'resnet50':
            # Bottleneck numbers [3, 4, 6, 3] for layer 1, 2, 3, 4
            self.resnet = resnet50(pretrained=True)
            self.encoder_list = nn.ModuleList([
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu,
                self.resnet.maxpool,
                self.resnet.layer1,
                self.resnet.layer2._modules['0'],  # layer2, bottleneck 0
                self.resnet.layer2._modules['1'],  # layer2, bottleneck 1
                self.resnet.layer2._modules['2']   # layer2, bottleneck 2, feature size B x 512 x 46 x 46, 1/8 (368x368)
                # self.resnet.layer2,
                # self.resnet.layer3
                # self.resnet.layer3._modules['0'],
                # self.resnet.layer3._modules['1'], # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
            ])
            self.block1 = nn.Sequential(
                self.resnet.layer2._modules['3'],
                self.resnet.layer3._modules['0'],
                self.resnet.layer3._modules['1'],  # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
                # nn.Conv2d(1024, 512, 1, 1, 0),  # feature size B x 512 x 23 x 23, 1/16 (368 x 368)
                # nn.ReLU(True)
            )
            # self.block2 = nn.Sequential(
            #     nn.Conv2d(1024, 2048, 5, 2, 2),  # 1/2
            #     nn.ReLU(True),
            #     nn.Conv2d(2048, 2048, 5, 1, 2),
            #     nn.ReLU(True),
            #     nn.Conv2d(2048, 2048, 5, 1, 2),
            #     nn.ReLU(True),
            #     nn.Conv2d(2048, 2048, 1, 1, 0),
            #     nn.ReLU(True),
            #     nn.Conv2d(2048, 1024, 1, 1, 0),  # feature size B x 1024 x 12 x 12
            #     nn.ReLU(True),
            # )
            self.block2 = nn.Sequential(
                self.resnet.layer3._modules['2'],
                self.resnet.layer3._modules['3'],
                self.resnet.layer3._modules['4'],
                self.resnet.layer3._modules['5'],
                self.resnet.layer4._modules['0'],
                self.resnet.layer4._modules['1'],  # feature size B x 2084 x 12 x 12

            )

            # self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.downsample = nn.Sequential(
                nn.Conv2d(512, 1024, 1, 2, 0),
                # nn.BatchNorm2d(1024),
                nn.ReLU(True)
            )
            # self.upsample = nn.UpsamplingNearest2d(size=(23, 23))
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, 3, 2, 1),
                nn.ReLU(True)
            )

            # self.channel_reduce = nn.Sequential(
            #     nn.Conv2d(1024*3, 1024, 1, 1, 0),
            #     # nn.BatchNorm2d(1024),
            #     nn.ReLU(True)
            # )

            # for i in range(5):
            #     self.block2[i*2].weight.data.normal_(mean=0, std=0.01)
            #     self.block2[i*2].weight.data.normal_(mean=0, std=0.01)
            self.downsample[0].weight.data.normal_(mean=0, std=0.01)
            self.upsample[0].weight.data.normal_(mean=0, std=0.01)
            # self.channel_reduce[0].weight.data.normal_(mean=0, std=0.01)

        print('Encoder initialized!')



    def forward(self, x: torch.Tensor):
        """feedforward propagation

        Arguments:
            x {torch.Tensor} -- input samples batch × 3 × height × width

        Returns:
            [torch.Tensor] --
        """
        # feature = self.encoder(input_data)

        for i in range(len(self.encoder_list)):
            x = self.encoder_list[i](x)

        x1 = self.block1(x)
        x2 = self.block2(x1)

        # x1 = self.upsample1(x1)
        x = self.downsample(x)
        x2 = self.upsample(x2)

        x = torch.cat([x, x1, x2], dim=1)
        # x = self.channel_reduce(x)
        # x = x + x1 + x2

        return x

class EncoderMultiScaleType2(nn.Module):
    def __init__(self, trunk='resnet50'):
        super(EncoderMultiScaleType2, self).__init__()

        if trunk == 'resnet50':
            # Bottleneck numbers [3, 4, 6, 3] for layer 1, 2, 3, 4
            self.resnet = resnet50(pretrained=True)
            self.encoder_list = nn.ModuleList([
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu,
                self.resnet.maxpool,
                self.resnet.layer1,
                self.resnet.layer2._modules['0'],  # layer2, bottleneck 0
                self.resnet.layer2._modules['1'],  # layer2, bottleneck 1
                self.resnet.layer2._modules['2']   # layer2, bottleneck 2, feature size B x 512 x 46 x 46, 1/8 (368x368)
                # self.resnet.layer2,
                # self.resnet.layer3
                # self.resnet.layer3._modules['0'],
                # self.resnet.layer3._modules['1'], # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
            ])
            self.block1 = nn.Sequential(
                self.resnet.layer2._modules['3'],
                self.resnet.layer3._modules['0'],
                self.resnet.layer3._modules['1'],  # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
                # nn.Conv2d(1024, 512, 1, 1, 0),  # feature size B x 512 x 23 x 23, 1/16 (368 x 368)
                # nn.ReLU(True)
            )
            # self.block2 = nn.Sequential(
            #     nn.Conv2d(1024, 2048, 5, 2, 2),  # 1/2
            #     nn.ReLU(True),
            #     nn.Conv2d(2048, 2048, 5, 1, 2),
            #     nn.ReLU(True),
            #     nn.Conv2d(2048, 2048, 5, 1, 2),
            #     nn.ReLU(True),
            #     nn.Conv2d(2048, 2048, 1, 1, 0),
            #     nn.ReLU(True),
            #     nn.Conv2d(2048, 1024, 1, 1, 0),  # feature size B x 1024 x 12 x 12
            #     nn.ReLU(True),
            # )
            self.block2 = nn.Sequential(
                self.resnet.layer3._modules['2'],
                self.resnet.layer3._modules['3'],
                self.resnet.layer3._modules['4'],
                self.resnet.layer3._modules['5'],
                self.resnet.layer4._modules['0'],
                self.resnet.layer4._modules['1'],  # feature size B x 2084 x 12 x 12

            )

            # self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.downsample = nn.Sequential(
                nn.Conv2d(512, 1024, 1, 2, 0),
                # nn.BatchNorm2d(1024),
                nn.ReLU(True)
            )
            # self.upsample = nn.UpsamplingNearest2d(size=(23, 23))
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, 3, 2, 1),
                nn.ReLU(True)
            )

            # self.channel_reduce = nn.Sequential(
            #     nn.Conv2d(1024*3, 1024, 1, 1, 0),
            #     # nn.BatchNorm2d(1024),
            #     nn.ReLU(True)
            # )

            # for i in range(5):
            #     self.block2[i*2].weight.data.normal_(mean=0, std=0.01)
            #     self.block2[i*2].weight.data.normal_(mean=0, std=0.01)
            self.downsample[0].weight.data.normal_(mean=0, std=0.01)
            self.upsample[0].weight.data.normal_(mean=0, std=0.01)
            # self.channel_reduce[0].weight.data.normal_(mean=0, std=0.01)

        print('Encoder initialized!')



    def forward(self, x_support: torch.Tensor, x_query: torch.Tensor, label_support: torch.Tensor, support_kp_mask: torch.Tensor):
        """feedforward propagation

        Arguments:
            x {torch.Tensor} -- input samples batch × 3 × height × width

        Returns:
            [torch.Tensor] --
        """
        for i in range(len(self.encoder_list)):
            x_support = self.encoder_list[i](x_support)
        x_support1 = self.block1(x_support)
        x_support2 = self.block2(x_support1)

        for i in range(len(self.encoder_list)):
            x_query = self.encoder_list[i](x_query)
        x_query1 = self.block1(x_query)
        x_query2 = self.block2(x_query1)

        (B, N, _) = label_support.shape  # B x N x 2
        C, C1, C2 = x_support.shape[1], x_support1.shape[1], x_support2.shape[1]  # B x C x H x W
        H, H1, H2 = x_support.shape[2], x_support1.shape[2], x_support2.shape[2]

        attentive_features, _ = feature_modulator2(x_support, label_support, support_kp_mask, x_query)  # B x N x 512 x 46 x 46
        attentive_features1, _ = feature_modulator2(x_support1, label_support, support_kp_mask, x_query1)  # B x N x 1024 x 23 x 23
        attentive_features2, _ = feature_modulator2(x_support2, label_support, support_kp_mask, x_query2)  # B x N x 2048 x 12 x 12

        attentive_features = attentive_features.reshape(B * N, C, H, -1)   # (B * N) x 512 x 46 x 46
        attentive_features2 = attentive_features2.reshape(B * N, C2, H2, -1)  # (B * N) x 2048 x 12 x 12

        # (B * N) x 512 x 46 x 46 --> (B * N) x 1024 x 23 x 23
        attentive_features = self.downsample(attentive_features)
        attentive_features = attentive_features.reshape(B, N, C1, H1, -1)  # B x N x 1024 x 23 x 23
        # (B * N) x 2048 x 12 x 12 --> (B * N) x 1024 x 23 x 23
        attentive_features2 = self.upsample(attentive_features2)
        attentive_features2 = attentive_features2.reshape(B, N, C1, H1, -1)  # B x N x 1024 x 23 x 23

        # B x N x (1024*3) x 23 x 23
        attentive_features1 = torch.cat([attentive_features, attentive_features1, attentive_features2], dim=2)
        # x = self.channel_reduce(x)
        # x = x + x1 + x2

        return attentive_features1


def feature_modulator(support_features, support_labels, support_kp_mask, query_features, context_mode='soft_fiber_bilinear', sigma=13,
                      downsize_factor=16, image_length=368, fused_attention=None, output_attention_maps=False):
    '''
    :param support_features: B1 x C x H' x W'
    :param support_labels:  B1 x N x 2
    :param query_features: B2 x C x H' x W'
    :return: modulated features (B2 x N x C x H' x W) and N attention maps (N x H' x W')
    '''
    # position attention
    # extract N keypoint embeddings from support features
    # modulate the B2 x C x H x W query features to B2 x N x C x H x W attentive features
    (B1, C, H, W) = support_features.shape
    N_categories = support_labels.shape[1]
    B2 = query_features.shape[0]

    # ====================
    if context_mode == 'hard_fiber':
        # method 1, hard quantization
        # scale the label (x, y) which ranges 0~1 to 0~image_width-1
        support_labels = support_labels.mul(W - 1).round().long().clamp(0, W - 1)
        support_embeddings = []
        for i in range(B1):
            embeddings_per_shot = support_features[i, :, support_labels[i, :, 1], support_labels[i, :, 0]]  # C x N
            support_embeddings.extend([embeddings_per_shot])
        support_features = torch.stack(support_embeddings)  # B1 x C x N
    # ====================
    elif context_mode == 'soft_fiber_bilinear':
        # method 2, bilinear interpolation and sampling
        # fsupport_labels = support_labels * (W - 1)  # floating-point coordinates
        #
        # isupport_labels = torch.floor(fsupport_labels)
        # offset_topleft = fsupport_labels - isupport_labels
        # x_offset, y_offset = offset_topleft[:, :, 0], offset_topleft[:, :, 1]
        # isupport_labels = isupport_labels.long()
        # x_topleft, y_topleft = isupport_labels[:, :, 0], isupport_labels[:, :, 1]  # B1 x N
        # x_br, y_br = x_topleft + 1, y_topleft + 1  # B1 x N
        # # we are careful to the boundary
        # is_topleft_x_in = (x_topleft >= 0) * (x_topleft <= W - 1)  # B1 x N
        # is_topleft_y_in = (y_topleft >= 0) * (y_topleft <= W - 1)  # B1 x N
        # is_bottomright_x_in = (x_br >= 0) * (x_br <= W - 1)        # B1 x N
        # is_bottomright_y_in = (y_br >= 0) * (y_br <= W - 1)        # B1 x N
        # x_topleft[~is_topleft_x_in] = 0
        # y_topleft[~is_topleft_y_in] = 0
        # x_br[~is_bottomright_x_in] = 0
        # y_br[~is_bottomright_y_in] = 0
        # flag_topleft = is_topleft_x_in * is_topleft_y_in       # B1 x N
        # flag_topright = is_bottomright_x_in * is_topleft_y_in  # B1 x N
        # flag_bl = is_topleft_x_in * is_bottomright_y_in        # B1 x N
        # flag_br = is_bottomright_x_in * is_bottomright_y_in    # B1 x N
        # support_embeddings = []
        # for i in range(B1):
        #     # bilinear interpolation
        #     # vi = (1-dx)*(1-dy)*f(p_topleft) + dx * (1-dy)*f(p_topright) + (1-dx)*dy*f(p_bottomleft) + dx*dy*f(p_bottomright)
        #     w_tl = flag_topleft[i, :] * (1 - x_offset[i, :]) * (1 - y_offset[i, :])
        #     w_tr = flag_topright[i, :] * x_offset[i, :] * (1 - y_offset[i, :])
        #     w_bl = flag_bl[i, :] * (1 - x_offset[i, :]) * y_offset[i, :]
        #     w_br = flag_br[i, :] * x_offset[i, :] * y_offset[i, :]
        #     # print_weights(w_tl)
        #     # print_weights(support_features[i, :, y_topleft[i, :], x_topleft[i, :]])
        #     # print_weights(w_tl * support_features[i, :, y_topleft[i, :], x_topleft[i, :]])
        #     embeddings_per_shot = w_tl * support_features[i, :, y_topleft[i, :], x_topleft[i, :]]  + \
        #         w_tr * support_features[i, :, y_topleft[i, :], x_br[i, :]] +\
        #         w_bl * support_features[i, :, y_br[i, :], x_topleft[i, :]] +\
        #         w_br * support_features[i, :, y_br[i, :], x_br[i, :]]
        #     support_embeddings.extend([embeddings_per_shot])
        #
        # support_features = torch.stack(support_embeddings)  # B1 x C x N

        # support_labels_normalized = ((support_labels - 0.5) * 2).reshape(B1, 1, N_categories, 2)  # ranges from -1 ~ 1, 4D tensor, which is the format for grid_sample
        support_labels_normalized = support_labels.reshape(B1, 1, N_categories, 2)
        support_features = nn.functional.grid_sample(support_features, support_labels_normalized, mode='bilinear', align_corners=True).reshape(B1, C, N_categories)  # B1 x C x N
        # ====================
    elif context_mode == 'soft_fiber_gaussian':
        # compute heatmap which is subject to Gaussian distribution
        heatmaps = torch.zeros(B1, N_categories, H, W).cuda()
        original_width = image_length  # 368
        sigma = sigma   # 14, 7
        stride = 16  # image width / W = image_height / H = 368 / 23 = 16
        # print_weights(support_labels*368)
        # print_weights(support_kp_mask)

        support_labels_normalized = support_labels / 2 + 0.5  # note that putGaussianMaps() requires label ranges 0~1
        for sample_i in range(B1):
            for kp_j in range(N_categories):
                if support_kp_mask[sample_i, kp_j] == 0:
                    continue
                # print(sample_i, kp_j)
                heatmaps[sample_i, kp_j, :, :], return_flag = putGaussianMaps(support_labels_normalized[sample_i, kp_j, :] * (original_width-1), sigma, H, W, stride, normalization=True)
                # if return_flag == False:
                #     print_weights(support_kp_mask)
                #     print_weights(support_labels)
                #     print(sample_i, kp_j)
                #     return  None, None
                #     exit(0)
        # compute fibers using Gaussian pooling
        support_embeddings = []
        for i in range(N_categories):
            kp_heatmap = heatmaps[:, i, :, :]
            kp_heatmap = kp_heatmap.view(B1, 1, H, W)
            # Using broadcast
            embeddings_per_category = torch.sum((support_features * kp_heatmap).reshape(B1, C, H*W), dim=2)  # B1 x C
            support_embeddings.extend([embeddings_per_category])
        support_features = torch.stack(support_embeddings)    # N x B1 x C
        support_features = support_features.permute(1, 2, 0)  # B1 x C x N
    elif context_mode == 'soft_fiber_gaussian2':  # gaussian pooling for both support and query features
        # original_width = 368
        sigma = sigma  # 14, 7
        stride = downsize_factor  # 16 or 32, image width / W = image_height / H = 368 / 23 = 16
        start = stride / 2.0 - 0.5
        kernel_w_half = round(3*sigma / stride)
        kernel_w = 2 * kernel_w_half + 1
        kernel_center= torch.Tensor([0.5, 0.5]).mul(kernel_w-1).cuda()
        gaussian_kernel = torch.zeros(kernel_w, kernel_w).cuda()  # gaussian kernel
        gaussian_kernel[:, :], return_flag = putGaussianMaps(kernel_center*stride + start, sigma, kernel_w, kernel_w, stride, normalization=True)
        gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_w, kernel_w)  # 1 x 1 x kernel_w x kernel_w
        # gaussian pooling for support and query features
        support_features = support_features.reshape(B1*C, 1, H, W)
        query_features = query_features.reshape(B2*C, 1, H, W)
        support_features = nn.functional.conv2d(support_features, gaussian_kernel, bias=None, stride=1, padding=kernel_w_half)
        query_features = nn.functional.conv2d(query_features, gaussian_kernel, bias=None, stride=1, padding=kernel_w_half)
        support_features = support_features.reshape(B1, C, H, W)
        query_features = query_features.reshape(B2, C, H, W)
        # compute support keypoint fibers, using bilinear interpolation
        # support_labels_normalized = ((support_labels - 0.5) * 2).reshape(B1, 1, N_categories, 2)  # ranges from -1 ~ 1, 4D tensor, which is the format for grid_sample
        support_labels_normalized = support_labels.reshape(B1, 1, N_categories, 2)
        support_features = nn.functional.grid_sample(support_features, support_labels_normalized, mode='bilinear', align_corners=True).reshape(B1, C, N_categories)  # B1 x C x N

    # ====================
    # method 1
    # #  average keypoint embeddings of each category for one episode
    # support_features2 = support_features.mean(dim=0)  # C x N
    # ====================
    # method 2
    # using support keypoint mask to compute the weighted support embeddings across shots
    support_kp_weight = torch.sum(support_kp_mask, dim=0)  # N
    for i in range(N_categories):
        if support_kp_weight[i] > 0:
            support_kp_weight[i] = 1 / support_kp_weight[i]
        else:  # equal to 0
            support_kp_weight[i] = 1
    mask = support_kp_mask.reshape(B1, 1, N_categories)  # B1 x 1 x N
    # using broadcast
    support_features = support_features * mask  # B1 x C x N
    support_features = torch.sum(support_features, dim=0)  # C x N
    # normalization by multiplying weights, using broadcast
    support_features = support_features * support_kp_weight.view(1, N_categories)  # C x N
    # ====================

    query_features_temp = query_features

    B2 = query_features.shape[0]
    query_features = query_features.reshape(B2, C, H * W )   # B2 x C x (H*W)
    # print(query_features.cpu().detach().numpy()[0])
    query_features_t = query_features.permute(0, 2, 1)  # B2 x (H*W) x C
    attention_maps = query_features_t.matmul(support_features)  # B2 x (H*W) x N

    if fused_attention == 'L2':
        # l2 norm
        query_features_l2 = query_features_temp * query_features_temp
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W)
        attention_maps_l2 = attention_maps / query_features_l2.view(B2, H * W, 1)
        # softmax
        attention_maps_l2 = nn.functional.softmax(attention_maps_l2, dim=1)    # B2 x (H*W) x N
        attention_maps_t = attention_maps_l2.permute(2, 0, 1)  # N x B2 x (H*W)
    elif fused_attention == 'L1':
        # l1 norm
        query_features_l1 = torch.abs(query_features_temp)
        query_features_l1 = query_features_l1.sum(dim=1).reshape(B2, H * W)
        attention_maps_l1 = attention_maps / query_features_l1.view(B2, H * W, 1)
        # softmax
        attention_maps_l1 = nn.functional.softmax(attention_maps_l1, dim=1)  # B2 x (H*W) x N
        attention_maps_t = attention_maps_l1.permute(2, 0, 1)  # N x B2 x (H*W)
    else:  # None, no norm is applied
        attention_maps_original = nn.functional.softmax(attention_maps, dim=1)
        attention_maps_t = attention_maps_original.permute(2, 0, 1)  # N x B2 x (H*W)

    if output_attention_maps == True:
        # l2 norm
        query_features_l2 = query_features_temp * query_features_temp
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W)
        attention_maps_l2 = attention_maps / query_features_l2.view(B2, H * W, 1)
        # softmax
        attention_maps_l2 = nn.functional.softmax(attention_maps_l2, dim=1)  # B2 x (H*W) x N
        attention_maps_l2 = attention_maps_l2.permute(0, 2, 1).reshape(B2, N_categories, H, W)

        # l1 norm
        query_features_l1 = torch.abs(query_features_temp)
        query_features_l1 = query_features_l1.sum(dim=1).reshape(B2, H * W)
        attention_maps_l1 = attention_maps / query_features_l1.view(B2, H * W, 1)
        # softmax
        attention_maps_l1 = nn.functional.softmax(attention_maps_l1, dim=1)  # B2 x (H*W) x N
        attention_maps_l1 = attention_maps_l1.permute(0, 2, 1).reshape(B2, N_categories, H, W)

        # copy attention maps, tensor, B2 x N x H x W
        attention_maps_l2_cpu = copy.deepcopy(attention_maps_l2.cpu().detach())
        attention_maps_l1_cpu = copy.deepcopy(attention_maps_l1.cpu().detach())
    else:
        attention_maps_l2_cpu, attention_maps_l1_cpu = None, None


    attention_maps_t = attention_maps_t.unsqueeze(dim=2)  # N x B2 x 1 x (H*W)

    attention_maps_t = attention_maps_t.repeat(1, 1, C, 1)  # N x B2 x C x (H*W)
    # for i in range(N_categories):
    #     attention_maps_t[i] = attention_maps_t[i].mul(query_features)
    #     attention_maps_t[i, :, :, :] = attention_maps_t[i, :, :, :].mul(query_features)
    query_features = query_features.unsqueeze(dim=0).repeat(N_categories, 1, 1, 1)  # N x B2 x C x (H*W)
    # attention_maps_t = attention_maps_t.detach()
    query_features = query_features * attention_maps_t  # N x B2 x C x (H*W)


    # B2 x N x C x (H*W)
    query_features = query_features.permute(1, 0, 2, 3)
    # B2 x N x C x H x W
    query_features = query_features.reshape(B2, N_categories, C, H, W)

    return query_features, attention_maps_l2_cpu, attention_maps_l1_cpu


def extract_representations(features, labels, kp_mask, context_mode='soft_fiber_bilinear', sigma=13, downsize_factor=16, image_length=368, together_trans_features=None):
    '''
    :param features: B1 x C x H' x W'
    :param labels: B1 x N x 2
    :param kp_mask: B1 x N
    :param context_mode:
    :param sigma:
    :param downsize_factor:
    :param image_length:
    :return: representations (B1 x C x N)
    '''
    (B1, C, H, W) = features.shape
    N_categories = labels.shape[1]
    # ====================
    if context_mode == 'hard_fiber':
        # method 1, hard quantization
        # scale the label (x, y) which ranges 0~1 to 0~image_width-1
        labels = labels.mul(W - 1).round().long().clamp(0, W - 1)
        embeddings = []
        for i in range(B1):
            embeddings_per_shot = features[i, :, labels[i, :, 1], labels[i, :, 0]]  # C x N
            embeddings.extend([embeddings_per_shot])
        representations = torch.stack(embeddings)  # B1 x C x N
    # ====================
    elif context_mode == 'soft_fiber_bilinear':
        # support_labels_normalized = ((support_labels - 0.5) * 2).reshape(B1, 1, N_categories, 2)  # ranges from -1 ~ 1, 4D tensor, which is the format for grid_sample
        labels_normalized = labels.reshape(B1, 1, N_categories, 2)
        representations = nn.functional.grid_sample(features, labels_normalized, mode='bilinear', align_corners=True).reshape(B1, C, N_categories)  # B1 x C x N
        # ====================
    elif context_mode == 'soft_fiber_gaussian':
        # compute heatmap which is subject to Gaussian distribution
        heatmaps = torch.zeros(B1, N_categories, H, W).cuda()
        original_width = image_length  # 368
        sigma = sigma   # 14, 7
        stride = downsize_factor  # 16 or 32, image width / W = image_height / H = 368 / 23 = 16
        # print_weights(support_labels*368)
        # print_weights(support_kp_mask)

        labels_normalized = labels / 2 + 0.5  # note that putGaussianMaps() requires label ranges 0~1
        for sample_i in range(B1):
            for kp_j in range(N_categories):
                if kp_mask[sample_i, kp_j] == 0:
                    continue
                # print(sample_i, kp_j)
                heatmaps[sample_i, kp_j, :, :], return_flag = putGaussianMaps(labels_normalized[sample_i, kp_j, :] * (original_width-1), sigma, H, W, stride, normalization=True)
                # if return_flag == False:
                #     print_weights(support_kp_mask)
                #     print_weights(support_labels)
                #     print(sample_i, kp_j)
                #     exit(0)
        # compute fibers using Gaussian pooling
        embeddings = []
        for i in range(N_categories):
            kp_heatmap = heatmaps[:, i, :, :]
            kp_heatmap = kp_heatmap.view(B1, 1, H, W)
            # Using broadcast
            embeddings_per_category = torch.sum((features * kp_heatmap).reshape(B1, C, H*W), dim=2)  # B1 x C
            embeddings.extend([embeddings_per_category])
        representations = torch.stack(embeddings)    # N x B1 x C
        representations = representations.permute(1, 2, 0)  # B1 x C x N
    elif context_mode == 'soft_fiber_gaussian2':  # gaussian pooling for both support and query features
        # original_width = 368
        sigma = sigma  # 14, 7
        stride = downsize_factor  # 16 or 32, image width / W = image_height / H = 368 / 23 = 16
        start = stride / 2.0 - 0.5
        kernel_w_half =  round(3*sigma / stride)
        kernel_w = 2 * kernel_w_half + 1
        kernel_center= torch.Tensor([0.5, 0.5]).mul(kernel_w-1).cuda()
        gaussian_kernel = torch.zeros(kernel_w, kernel_w).cuda()  # gaussian kernel
        gaussian_kernel[:, :], return_flag = putGaussianMaps(kernel_center*stride + start, sigma, kernel_w, kernel_w, stride, normalization=True)
        # temp = gaussian_kernel.cpu().detach().numpy()
        gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_w, kernel_w)  # 1 x 1 x kernel_w x kernel_w
        # gaussian pooling for support and query features
        representations = features.reshape(B1*C, 1, H, W)
        representations = nn.functional.conv2d(representations, gaussian_kernel, bias=None, stride=1, padding=kernel_w_half)
        representations = representations.reshape(B1, C, H, W)

        #===========
        if together_trans_features is not None:
            B2 = together_trans_features.shape[0]
            conv_query_features = together_trans_features.reshape(B2*C, 1, H, W)
            conv_query_features = nn.functional.conv2d(conv_query_features, gaussian_kernel, bias=None, stride=1, padding=kernel_w_half)
            conv_query_features = conv_query_features.reshape(B2, C, H, W)
        #===========

        # compute support keypoint fibers, using bilinear interpolation
        # support_labels_normalized = ((support_labels - 0.5) * 2).reshape(B1, 1, N_categories, 2)  # ranges from -1 ~ 1, 4D tensor, which is the format for grid_sample
        labels_normalized = labels.reshape(B1, 1, N_categories, 2)
        representations = nn.functional.grid_sample(representations, labels_normalized, mode='bilinear', align_corners=True).reshape(B1, C, N_categories)  # B1 x C x N

    if kp_mask is not None:
        mask = kp_mask.reshape(B1, 1, N_categories)  # B1 x 1 x N
        # using broadcast
        representations = representations * mask  # B1 x C x N

    # print_weights(representations)

    if together_trans_features is not None:
        return representations, conv_query_features  # also applied the Gaussian conv to query_feature
    else:
        return representations, None

def average_representations(kp_representations: torch.Tensor, kp_mask: torch.Tensor):
    '''
    compute keypoint proto-types
    :param kp_representations: B x C x N
    :param kp_mask: B x N
    :return: avg_representations, C x N
    '''
    B, N_categories = kp_mask.shape

    # ====================
    # method 1
    # #  average keypoint embeddings of each category for one episode
    # avg_representations = kp_representations.mean(dim=0)  # C x N
    # ====================
    # method 2
    # using support keypoint mask to compute the weighted support embeddings across shots
    kp_weight = torch.sum(kp_mask, dim=0)  # N
    for i in range(N_categories):
        if kp_weight[i] > 0:
            kp_weight[i] = kp_weight[i].reciprocal()
        else:  # equal to 0
            kp_weight[i] = 1
    mask = kp_mask.reshape(B, 1, N_categories)  # B1 x 1 x N
    # using broadcast
    kp_representations2 = kp_representations * mask  # B1 x C x N
    # print_weights(kp_representations2)
    sum_representations = torch.sum(kp_representations2, dim=0)  # C x N
    # normalization by multiplying weights, using broadcast
    avg_representations = sum_representations * kp_weight.view(1, N_categories)  # C x N
    # ====================

    return avg_representations

def average_representations2(kp_representations: torch.Tensor, kp_mask: torch.Tensor):
    '''
    compute keypoint proto-types
    :param kp_representations: B x C x N
    :param kp_mask: B x N
    :return: avg_representations, C x N
    '''

    B, N_categories = kp_mask.shape
    kp_mask_temp = torch.clone(kp_mask)

    # ====================
    # method 1
    # #  average keypoint embeddings of each category for one episode
    # avg_representations = kp_representations.mean(dim=0)  # C x N
    # ====================
    # method 2
    # using support keypoint mask to compute the weighted support embeddings across shots
    kp_weight = torch.sum(kp_mask_temp, dim=0)  # N
    for i in range(N_categories):
        if kp_weight[i] > 0:
            kp_weight[i] = kp_weight[i].reciprocal()  # 1.0 / B
        else:  # equal to 0
            kp_weight[i] = 1.0 / B
            kp_mask_temp[:, i] = 1.0    # in order to avoid zero for final summation
    mask = kp_mask_temp.reshape(B, 1, N_categories)  # B1 x 1 x N
    # using broadcast
    kp_representations2 = kp_representations * mask  # B1 x C x N
    # print_weights(kp_representations2)
    sum_representations = torch.sum(kp_representations2, dim=0)  # C x N
    # normalization by multiplying weights, using broadcast
    avg_representations = sum_representations * kp_weight.view(1, N_categories)  # C x N
    # ====================

    return avg_representations

def average_representations3(kp_representations: torch.Tensor, kp_mask: torch.Tensor):
    '''
    compute keypoint proto-types
    :param kp_representations: B x C x N
    :param kp_mask: B x N
    :return: avg_representations, C x N
    '''

    B, N_categories = kp_mask.shape
    kp_mask_temp = torch.clone(kp_mask)

    # ====================
    # method 1
    # #  average keypoint embeddings of each category for one episode
    # avg_representations = kp_representations.mean(dim=0)  # C x N
    # ====================
    # method 2
    # using support keypoint mask to compute the weighted support embeddings across shots
    kp_weight = torch.sum(kp_mask_temp, dim=0)  # N
    for i in range(N_categories):
        if kp_weight[i] > 0:
            kp_weight[i] = 1.0 / B # kp_weight[i].reciprocal()
        else:  # equal to 0
            kp_weight[i] = 1.0 / B
            kp_mask_temp[:, i] = 1.0    # in order to avoid zero for final summation
    mask = kp_mask_temp.reshape(B, 1, N_categories)  # B1 x 1 x N
    # using broadcast
    kp_representations2 = kp_representations * mask  # B1 x C x N
    # print_weights(kp_representations2)
    sum_representations = torch.sum(kp_representations2, dim=0)  # C x N
    # normalization by multiplying weights, using broadcast
    avg_representations = sum_representations * kp_weight.view(1, N_categories)  # C x N
    # ====================

    return avg_representations

def compute_similarity_map(support_fibers, query_features, attentive_features):
    '''
    compute normed similarity map; please note that we do not apply softmax over image space (H * W)
    :param support_fibers: C x N
    :param query_features: B2 x C x H x W
    :param attentive_features: B2 x N x C x H x W
    :return: similarity_map, B2 x N x H x W
    '''
    B2, N_categories, C, H, W = attentive_features.shape
    query_features_l2 = query_features **2  # B x C x H x W
    query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W).detach()  # B2 x (H*W)

    support_features_l2 = support_fibers * support_fibers  # C x N
    support_features_l2 = torch.sqrt(support_features_l2.sum(dim=0)).detach()  # N

    # norm = query_features_l2.unsqueeze(dim=1).repeat(1, N_categories, 1) * support_features_l2.view(1, N_categories, 1)  # B2 x N x (H*W)
    norm = query_features_l2.unsqueeze(dim=1) * support_features_l2.view(1, N_categories, 1)  # B2 x N x (H*W)
    # print('simi: ', torch.sum((norm-norm2)**2))

    attention_maps = attentive_features.sum(dim=2).reshape(B2, N_categories, H * W)

    similarity_map = (attention_maps / (norm + 1e-9)).reshape(B2, N_categories, H, W)  # B x N x H x W


    return similarity_map

def feature_modulator3(support_fibers, query_features, fused_attention='L2-c', output_attention_maps=False, compute_similarity=False):
    '''
    compute attentive features
    :param support_fibers: C x N, proto-types of support keypoints
    :param query_features: B2 x C x H x W
    :param fused_attention:
    :param output_attention_maps:
    :param compute_similarity:
    :return: modulated features (B2 x N x C x H' x W') and N attention maps (N x H' x W')
    # channel-wise mulplication
    # extract N keypoint embeddings from support features
    # modulate the B2 x C x H x W query features to B2 x N x C x H x W attentive features
    '''
    N_categories = support_fibers.shape[1]
    B2, C, H, W = query_features.shape

    # query_features_temp = query_features     # B2 x C x H x W
    # support_features_temp = support_fibers # C x N

    # print_weights(support_fibers)
    # print_weights(query_features)

    # # compute method 1
    # support_features = support_fibers.permute(1, 0).unsqueeze(dim=2).repeat(1, 1, H * W)  # N x C x (H*W)
    # support_features = support_features.unsqueeze(dim=0).repeat(B2, 1, 1, 1)  # B2 x N x C x (H*W)
    # attentive_features = query_features.reshape(B2, C, H * W)  # B2 x C x (H*W)
    # attentive_features = attentive_features.unsqueeze(dim=1).repeat(1, N_categories, 1, 1)  # B2 x N x C x (H*W)
    # attentive_features = attentive_features * support_features  # B2 x N x C x (H*W)
    # # modulated attentive features, B2 x N x C x H x W
    # attentive_features = attentive_features.reshape(B2, N_categories, C, H, W)

    # compute method2, using broadcast
    attentive_features = query_features.reshape(B2, C, H*W).unsqueeze(dim=1) * support_fibers.permute(1,0).reshape(1, N_categories, C, 1)  # B2 x N x C x (H*W)
    attentive_features = attentive_features.reshape(B2, N_categories, C, H, W)  # modulated attentive features, B2 x N x C x H x W
    # print("atten: ", torch.sum((attentive_features-attentive_features2)**2))


    if compute_similarity == True:
        # note that here the query_features are attentive features, normed_similarity_map are l2 normed similarity map
        # normed_similarity_map = compute_similarity_map(support_fibers, query_features, attentive_features)
        support_fibers_L2 = torch.norm(support_fibers, dim=0)
        query_features_L2 = torch.norm(query_features, dim=1)
        inner_product = (query_features.reshape(B2, C, 1, H, W) * support_fibers.reshape(1, C, N_categories, 1, 1)).sum(dim=1)
        normed_similarity_map = inner_product / (query_features_L2.reshape(B2, 1, H, W) * support_fibers_L2.reshape(1, N_categories, 1, 1) + 1e-6)
        normed_similarity_map = torch.clamp(normed_similarity_map, 0, 1)

        assert torch.max(normed_similarity_map) <= 1.0, f"Maximum Error, Got max={torch.max(normed_similarity_map)}"
        assert torch.min(normed_similarity_map) >= 0.0, f"Minimum Error, Got min={torch.min(normed_similarity_map)}"

    else:
        normed_similarity_map = None

    if (fused_attention is not None) or (output_attention_maps == True):
        # it should be copied in avoid to modify query_features
        # method 1: mean --> softmax
        # attention_maps = query_features.mean(dim=2).reshape(B2, N_categories, H*W)
        # method 2: max  --> softmax
        # attention_maps = query_features.max(dim=2)[0].reshape(B2, N_categories, H * W)
        # method 3: sum --> softmax
        sum_attentive_features = attentive_features.sum(dim=2).reshape(B2, N_categories, H * W)  # note this is collapsed attentive features
        # method 4: sum --> relu --> softmax
        # attention_maps = nn.functional.relu(query_features.sum(dim=2)).reshape(B2, N_categories, H * W)
        # attention_maps = nn.functional.softmax(attention_maps, dim=2).reshape(B2, N_categories, H, W)
        # attention_maps_np = attention_maps.cpu().detach().numpy().copy()


    if fused_attention == 'L2-a':
        # l2 norm, spatial attention, applied to query features
        query_features_l2 = query_features * query_features
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W)
        attention_maps_l2 = sum_attentive_features / query_features_l2.view(B2, 1, H * W)
        # softmax
        # attention_maps_l2 = nn.functional.softmax(attention_maps_l2, dim=2)     # B2 x N x (H*W)
        attention_maps_l2 = attention_maps_l2.reshape(B2, N_categories, H, W)   # B2 x N x H x W

        # fuse space-channel attention
        # attention_maps_l2 = attention_maps_l2.detach()

        # here the query feature is attentive feature
        attentive_features *= attention_maps_l2.view(B2, N_categories, 1, H, W)
    elif fused_attention == 'L2-b':
        # l2 norm, fuse spatial and channel attention, applied to query features
        query_features_l2 = query_features * query_features  # B x C x H x W
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W)  # B2 x (H*W)

        support_features_l2 = support_fibers * support_fibers  # C x N
        support_features_l2 = torch.sqrt(support_features_l2.sum(dim=0))  # N

        # norm = torch.matmul(query_features_l2.view(B2, H*W, 1), support_features_l2.view(1, N_categories)).permute(0, 2, 1)  # B2 x N x (H*W)
        norm = query_features_l2.unsqueeze(dim=1).repeat(1, N_categories, 1) * support_features_l2.view(1, N_categories, 1)  # B2 x N x (H*W)

        # here the query feature is attentive feature
        attentive_features /= (norm.reshape(B2, N_categories, 1, H, W) + 1e-9)   # B2 x N x C x H x W
    elif fused_attention == 'L2-c':
        # l2 norm, using spatial attentiom map directly
        query_features_l2 = query_features * query_features  # B x C x H x W
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W).detach()  # B2 x (H*W)

        # support_features_l2 = support_fibers * support_fibers  # C x N
        # support_features_l2 = torch.sqrt(support_features_l2.sum(dim=0)).detach()  # N

        # norm = torch.matmul(query_features_l2.view(B2, H*W, 1), support_features_l2.view(1, N_categories)).permute(0, 2, 1).detach()  # B2 x N x (H*W)
        # norm = query_features_l2.unsqueeze(dim=1).repeat(1, N_categories, 1) * support_features_l2.view(1, N_categories, 1)  # B2 x N x (H*W)

        # attentive_features = (sum_attentive_features / (norm + 1e-9)).reshape(B2, N_categories, 1, H, W)   # B2 x N x 1 x H x W
        # attentive_features = (sum_attentive_features / (query_features_l2 + 1e-9).view(B2, 1, H*W)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
        # attentive_features = (sum_attentive_features / (support_features_l2 + 1e-9).unsqueeze(dim=0).repeat(B2, 1).view(B2, N_categories, 1)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W

        attentive_features = (sum_attentive_features / (query_features_l2 + 1e-9).view(B2, 1, H*W)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
        # attentive_features = (sum_attentive_features / (support_features_l2 + 1e-9).unsqueeze(dim=0).repeat(B2, 1).view(B2, N_categories, 1)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
    elif fused_attention == 'L2-d':
        # l2 norm, using spatial attentiom map directly
        query_features_l2 = query_features * query_features  # B x C x H x W
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W).detach()  # B2 x (H*W)

        support_features_l2 = support_fibers * support_fibers  # C x N
        support_features_l2 = torch.sqrt(support_features_l2.sum(dim=0)).detach()  # N

        # norm = torch.matmul(query_features_l2.view(B2, H*W, 1), support_features_l2.view(1, N_categories)).permute(0, 2, 1).detach()  # B2 x N x (H*W)
        norm = query_features_l2.unsqueeze(dim=1).repeat(1, N_categories, 1) * support_features_l2.view(1, N_categories, 1)  # B2 x N x (H*W)

        attentive_features = (sum_attentive_features / (norm + 1e-9)).reshape(B2, N_categories, 1, H, W)   # B2 x N x 1 x H x W
        # attentive_features = (sum_attentive_features / (query_features_l2 + 1e-9).view(B2, 1, H*W)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
        # attentive_features = (sum_attentive_features / (support_features_l2 + 1e-9).unsqueeze(dim=0).repeat(B2, 1).view(B2, N_categories, 1)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W

        # attentive_features = (sum_attentive_features / (query_features_l2 + 1e-9).view(B2, 1, H*W)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
        # attentive_features = (sum_attentive_features / (support_features_l2 + 1e-9).unsqueeze(dim=0).repeat(B2, 1).view(B2, N_categories, 1)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
    elif fused_attention == 'L1-a':
        # l1 norm, spatial attention, applied to query features
        query_features_l1 = torch.abs(query_features)
        query_features_l1 = query_features_l1.sum(dim=1).reshape(B2, H * W)
        attention_maps_l1 = sum_attentive_features / query_features_l1.view(B2, 1, H * W)
        # softmax
        # attention_maps_l1 = nn.functional.softmax(attention_maps_l1, dim=2)    # B2 x N x (H*W)
        attention_maps_l1 = attention_maps_l1.reshape(B2, N_categories, H, W)  # B2 x N x H x W

        # fuse space-channel attention
        # attention_maps_l1 = attention_maps_l1.detach()
        # here the query feature is attentive feature
        attentive_features *= attention_maps_l1.view(B2, N_categories, 1, H, W)
    elif fused_attention == 'attmap':
        # use attention map directly, average by number of feature channel
        attentive_features = (sum_attentive_features / C).reshape(B2, N_categories, 1, H, W)
    else:  # None
        pass

    if output_attention_maps == True:
        # l2 norm
        query_features_l2 = query_features * query_features
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W)
        attention_maps_l2 = sum_attentive_features / query_features_l2.view(B2, 1, H * W)

        # l2 norm but using support fibers
        # support_features_l2 = support_fibers * support_fibers  # C x N
        # support_features_l2 = torch.sqrt(support_features_l2.sum(dim=0))  # N
        # attention_maps_l2 = (sum_attentive_features / support_features_l2.unsqueeze(dim=0).repeat(B2, 1).view(B2, N_categories, 1)).reshape(B2, N_categories, 1, H, W)  # B2 x N x H x W

        # softmax
        attention_maps_l2 = nn.functional.softmax(attention_maps_l2, dim=2)    # B2 x N x (H*W)
        attention_maps_l2 = attention_maps_l2.reshape(B2, N_categories, H, W)  # B2 x N x H x W

        # l1 norm
        query_features_l1 = torch.abs(query_features)
        query_features_l1 = query_features_l1.sum(dim=1).reshape(B2, H * W)
        attention_maps_l1 = sum_attentive_features / query_features_l1.view(B2, 1, H * W)
        # softmax
        attention_maps_l1 = nn.functional.softmax(attention_maps_l1, dim=2)    # B2 x N x (H*W)
        attention_maps_l1 = attention_maps_l1.reshape(B2, N_categories, H, W)  # B2 x N x H x W

        # copy attention maps, tensor, B2 x N x H x W
        attention_maps_l2_cpu = copy.deepcopy(attention_maps_l2.cpu().detach())
        attention_maps_l1_cpu = copy.deepcopy(attention_maps_l1.cpu().detach())
    else:
        attention_maps_l2_cpu, attention_maps_l1_cpu = None, None


    return attentive_features, attention_maps_l2_cpu, attention_maps_l1_cpu, normed_similarity_map

def feature_modulator2(support_features, support_labels, support_kp_mask, query_features, context_mode='soft_fiber_bilinear', sigma=13,
                       downsize_factor=16, image_length=368, fused_attention='L2', output_attention_maps=False, compute_similarity=False):
    '''
    :param support_features: B1 x C x H' x W'
    :param support_labels:  B1 x N x 2
    :param support_kp_mask: B1 x N
    :param query_features: B2 x C x H' x W'
    :return: modulated features (B2 x N x C x H' x W') and N attention maps (N x H' x W')

    # channel-wise mulplication
    # extract N keypoint embeddings from support features
    # modulate the B2 x C x H x W query features to B2 x N x C x H x W attentive features
    '''
    # support_features = torch.linspace(1, 8, 8).reshape(1, 2, 2, 2).cuda()
    # support_labels = (torch.Tensor([[0, 0], [0, 1]]).float().cuda() - 0.5) * 2
    # support_kp_mask = torch.Tensor([[1, 1]]).cuda()
    # query_features = torch.linspace(1, 8, 8).reshape(1, 2, 2, 2).cuda()
    # print(support_features)
    # print(support_labels)
    # print(support_kp_mask)
    # print(query_features)

    (B1, C, H, W) = support_features.shape
    N_categories = support_labels.shape[1]
    B2 = query_features.shape[0]

    # ====================
    if context_mode == 'hard_fiber':
        # method 1, hard quantization
        # scale the label (x, y) which ranges 0~1 to 0~image_width-1
        support_labels = support_labels.mul(W - 1).round().long().clamp(0, W - 1)
        support_embeddings = []
        for i in range(B1):
            embeddings_per_shot = support_features[i, :, support_labels[i, :, 1], support_labels[i, :, 0]]  # C x N
            support_embeddings.extend([embeddings_per_shot])
        support_features = torch.stack(support_embeddings)  # B1 x C x N
    # ====================
    elif context_mode == 'soft_fiber_bilinear':
        # method 2, bilinear interpolation and sampling
        # fsupport_labels = support_labels * (W - 1)  # floating-point coordinates
        #
        # isupport_labels = torch.floor(fsupport_labels)
        # offset_topleft = fsupport_labels - isupport_labels
        # x_offset, y_offset = offset_topleft[:, :, 0], offset_topleft[:, :, 1]
        # isupport_labels = isupport_labels.long()
        # x_topleft, y_topleft = isupport_labels[:, :, 0], isupport_labels[:, :, 1]  # B1 x N
        # x_br, y_br = x_topleft + 1, y_topleft + 1  # B1 x N
        # # we are careful to the boundary
        # is_topleft_x_in = (x_topleft >= 0) * (x_topleft <= W - 1)  # B1 x N
        # is_topleft_y_in = (y_topleft >= 0) * (y_topleft <= W - 1)  # B1 x N
        # is_bottomright_x_in = (x_br >= 0) * (x_br <= W - 1)        # B1 x N
        # is_bottomright_y_in = (y_br >= 0) * (y_br <= W - 1)        # B1 x N
        # x_topleft[~is_topleft_x_in] = 0
        # y_topleft[~is_topleft_y_in] = 0
        # x_br[~is_bottomright_x_in] = 0
        # y_br[~is_bottomright_y_in] = 0
        # flag_topleft = is_topleft_x_in * is_topleft_y_in       # B1 x N
        # flag_topright = is_bottomright_x_in * is_topleft_y_in  # B1 x N
        # flag_bl = is_topleft_x_in * is_bottomright_y_in        # B1 x N
        # flag_br = is_bottomright_x_in * is_bottomright_y_in    # B1 x N
        # support_embeddings = []
        # for i in range(B1):
        #     # bilinear interpolation
        #     # vi = (1-dx)*(1-dy)*f(p_topleft) + dx * (1-dy)*f(p_topright) + (1-dx)*dy*f(p_bottomleft) + dx*dy*f(p_bottomright)
        #     w_tl = flag_topleft[i, :] * (1 - x_offset[i, :]) * (1 - y_offset[i, :])
        #     w_tr = flag_topright[i, :] * x_offset[i, :] * (1 - y_offset[i, :])
        #     w_bl = flag_bl[i, :] * (1 - x_offset[i, :]) * y_offset[i, :]
        #     w_br = flag_br[i, :] * x_offset[i, :] * y_offset[i, :]
        #     # print_weights(w_tl)
        #     # print_weights(support_features[i, :, y_topleft[i, :], x_topleft[i, :]])
        #     # print_weights(w_tl * support_features[i, :, y_topleft[i, :], x_topleft[i, :]])
        #     embeddings_per_shot = w_tl * support_features[i, :, y_topleft[i, :], x_topleft[i, :]]  + \
        #         w_tr * support_features[i, :, y_topleft[i, :], x_br[i, :]] +\
        #         w_bl * support_features[i, :, y_br[i, :], x_topleft[i, :]] +\
        #         w_br * support_features[i, :, y_br[i, :], x_br[i, :]]
        #     support_embeddings.extend([embeddings_per_shot])
        #
        # support_features = torch.stack(support_embeddings)  # B1 x C x N

        # support_labels_normalized = ((support_labels - 0.5) * 2).reshape(B1, 1, N_categories, 2)  # ranges from -1 ~ 1, 4D tensor, which is the format for grid_sample
        support_labels_normalized = support_labels.reshape(B1, 1, N_categories, 2)
        support_features = nn.functional.grid_sample(support_features, support_labels_normalized, mode='bilinear', align_corners=True).reshape(B1, C, N_categories)  # B1 x C x N
        # ====================
    elif context_mode == 'soft_fiber_gaussian':
        # compute heatmap which is subject to Gaussian distribution
        heatmaps = torch.zeros(B1, N_categories, H, W).cuda()
        original_width = image_length  # 368
        sigma = sigma   # 14, 7
        stride = downsize_factor  # 16 or 32, image width / W = image_height / H = 368 / 23 = 16
        # print_weights(support_labels*368)
        # print_weights(support_kp_mask)

        support_labels_normalized = support_labels / 2 + 0.5  # note that putGaussianMaps() requires label ranges 0~1
        for sample_i in range(B1):
            for kp_j in range(N_categories):
                if support_kp_mask[sample_i, kp_j] == 0:
                    continue
                # print(sample_i, kp_j)
                heatmaps[sample_i, kp_j, :, :], return_flag = putGaussianMaps(support_labels_normalized[sample_i, kp_j, :] * (original_width-1), sigma, H, W, stride, normalization=True)
                # if return_flag == False:
                #     print_weights(support_kp_mask)
                #     print_weights(support_labels)
                #     print(sample_i, kp_j)
                #     exit(0)
        # compute fibers using Gaussian pooling
        support_embeddings = []
        for i in range(N_categories):
            kp_heatmap = heatmaps[:, i, :, :]
            kp_heatmap = kp_heatmap.view(B1, 1, H, W)
            # Using broadcast
            embeddings_per_category = torch.sum((support_features * kp_heatmap).reshape(B1, C, H*W), dim=2)  # B1 x C
            support_embeddings.extend([embeddings_per_category])
        support_features = torch.stack(support_embeddings)    # N x B1 x C
        support_features = support_features.permute(1, 2, 0)  # B1 x C x N
    elif context_mode == 'soft_fiber_gaussian2':  # gaussian pooling for both support and query features
        # original_width = 368
        sigma = sigma  # 14, 7
        stride = downsize_factor  # 16 or 32, image width / W = image_height / H = 368 / 23 = 16
        start = stride / 2.0 - 0.5
        kernel_w_half =  round(3*sigma / stride)
        kernel_w = 2 * kernel_w_half + 1
        kernel_center= torch.Tensor([0.5, 0.5]).mul(kernel_w-1).cuda()
        gaussian_kernel = torch.zeros(kernel_w, kernel_w).cuda()  # gaussian kernel
        gaussian_kernel[:, :], return_flag = putGaussianMaps(kernel_center*stride + start, sigma, kernel_w, kernel_w, stride, normalization=True)
        # temp = gaussian_kernel.cpu().detach().numpy()
        gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_w, kernel_w)  # 1 x 1 x kernel_w x kernel_w
        # gaussian pooling for support and query features
        support_features = support_features.reshape(B1*C, 1, H, W)
        # query_features = query_features.reshape(B2*C, 1, H, W)
        support_features = nn.functional.conv2d(support_features, gaussian_kernel, bias=None, stride=1, padding=kernel_w_half)
        # query_features = nn.functional.conv2d(query_features, gaussian_kernel, bias=None, stride=1, padding=kernel_w_half)
        support_features = support_features.reshape(B1, C, H, W)
        # query_features = query_features.reshape(B2, C, H, W)

        # compute support keypoint fibers, using bilinear interpolation
        # support_labels_normalized = ((support_labels - 0.5) * 2).reshape(B1, 1, N_categories, 2)  # ranges from -1 ~ 1, 4D tensor, which is the format for grid_sample
        support_labels_normalized = support_labels.reshape(B1, 1, N_categories, 2)
        support_features = nn.functional.grid_sample(support_features, support_labels_normalized, mode='bilinear', align_corners=True).reshape(B1, C, N_categories)  # B1 x C x N

    elif context_mode == 'feature_conv':
        original_width = 368
        sigma = sigma  # 14, 7
        stride = 16  # image width / W = image_height / H = 368 / 23 = 16
        start = stride / 2.0 - 0.5
        kernel_w_half = round(3*sigma / stride)
        kernel_w = 2 * kernel_w_half + 1
        if kernel_w <= 0:
            print('error in feature_modulator2, kernel_w should > 0.')
        kernel_center= torch.Tensor([0.5, 0.5]).mul(kernel_w-1).cuda()
        gaussian_kernel = torch.zeros(kernel_w, kernel_w).cuda()  # gaussian kernel
        gaussian_kernel[:, :], return_flag = putGaussianMaps(kernel_center*stride + start, sigma, kernel_w, kernel_w, stride, normalization=True)
        # construct the grids, B1 x N x (kw x kw) x 2 (each spatial grid stores coordinate (x, y)), finally the (x, y) should normalize to -1~1
        grids = torch.zeros(B1, N_categories, kernel_w*kernel_w, 2).cuda()  # B1 x N x (kw*kw) x 2
        stride_normalized = stride / (original_width - 1)
        for sample_i in range(B1):
            for kp_j in range(N_categories):
                for k in range(kernel_w*kernel_w):
                    delta_y = k // kernel_w
                    delta_x = k % kernel_w
                    delta_y = (delta_y - kernel_w_half) * stride_normalized
                    delta_x = (delta_x - kernel_w_half) * stride_normalized
                    grids[sample_i, kp_j, k, 0] = support_labels[sample_i, kp_j, 0] + delta_x
                    grids[sample_i, kp_j, k, 1] = support_labels[sample_i, kp_j, 1] + delta_y
        # print_weights(support_labels)
        # print_weights(support_kp_mask)
        # print_weights(gaussian_kernel)
        # print_weights(grids)
        grids = (grids - 0.5) * 2  # ranges from -1 ~ 1, 4D tensor, which is the format for grid_sample
        support_features = nn.functional.grid_sample(support_features, grids, mode='bilinear', align_corners=True)  # B1 x C x N x (kw*kw)
        # gaussian weighting, using broadcasting
        support_features = support_features * gaussian_kernel.view(1, 1, 1, -1)  # B1 x C x N x (kw*kw)
        support_features = support_features.permute(0, 2, 1, 3)  # B1 x N x C x (kw*kw)

        # masking and average
        support_kp_weight = torch.sum(support_kp_mask, dim=0)  # N
        for i in range(N_categories):
            if support_kp_weight[i] > 0:
                support_kp_weight[i] = support_kp_weight[i].reciprocal()
            else:  # equal to 0
                support_kp_weight[i] = 1
        mask = support_kp_mask.reshape(B1, N_categories, 1, 1)  # B1 x N x 1 x 1
        # using broadcast
        support_features = support_features * mask  # B1 x N x C x (kw*kw)
        support_features = torch.sum(support_features, dim=0)  # N x C x (kw*kw)
        # normalization by multiplying weights, using broadcast
        support_features = support_features * support_kp_weight.reshape(N_categories, 1, 1)  # N x C x (kw*kw)

        # modulating query features
        # B2 x (C * kw * kw) x L, L = H*W which is the number of sliding local blocks
        query_features = torch.nn.functional.unfold(query_features, kernel_size=(kernel_w, kernel_w), padding=(kernel_w_half, kernel_w_half), stride=(1,1))
        query_features = query_features.reshape(B2, C, kernel_w*kernel_w, H*W)  # B2 x C x (kw*kw) x (H*W)

        # query_features = query_features.unsqueeze(dim=1).repeat(1, N_categories, 1, 1, 1)  # B2 x N x C x (kw*kw) x (H*W)
        # support_features = support_features.unsqueeze(dim=0).repeat(B2, 1, 1, 1).unsqueeze(dim=4) # B2 x N x C x (kw*kw) x 1
        # query_features = (query_features * support_features).sum(dim=3)  # using broadcast, B2 x N x C x (H*W)
        # query_features = query_features.reshape(B2, N_categories, C, H, W)

        modulated_features = []
        for i in range(N_categories):
            fiber_block = support_features[i, :, :]
            embeddings_per_category = query_features * fiber_block.view(1, C, kernel_w*kernel_w, 1)  # B2 x C x (kw*kw) x (H*W)
            embeddings_per_category = embeddings_per_category.sum(dim=2)
            modulated_features.extend([embeddings_per_category])
        query_features = torch.stack(modulated_features)  #  N x B2 x C x (H*W)
        query_features = query_features.permute(1, 0, 2, 3).reshape(B2, N_categories, C, H, W)

        attention_maps_np = None

        return query_features, attention_maps_np


    # avgs = average_representations(support_features, support_kp_mask)
    # print_weights(avgs)
    #====================
    # method 1
    # #  average keypoint embeddings of each category for one episode
    # support_features2 = support_features.mean(dim=0)  # C x N
    # ====================
    # method 2
    # using support keypoint mask to compute the weighted support embeddings across shots
    support_kp_weight = torch.sum(support_kp_mask, dim=0)  # N
    for i in range(N_categories):
        if support_kp_weight[i] > 0:
            support_kp_weight[i] = support_kp_weight[i].reciprocal()
        else:  # equal to 0
            support_kp_weight[i] = 1
    mask = support_kp_mask.reshape(B1, 1, N_categories)  # B1 x 1 x N
    # using broadcast
    support_features = support_features * mask  # B1 x C x N
    # print_weights(support_features)
    support_features = torch.sum(support_features, dim=0)  # C x N
    # normalization by multiplying weights, using broadcast
    support_features = support_features * support_kp_weight.view(1, N_categories)  # C x N
    # ====================
    # print_weights(support_features)
    # print_weights(query_features[0, :, 0, 0])
    query_features_temp = query_features     # B2 x C x H x W
    support_features_temp = support_features # C x N

    # print("feature difference: ", torch.sum(support_features - support_features2).cpu().detach())
    # H, W = 2, 2
    # C = 3
    # N_categories = 2
    # support_features = torch.zeros(3, 2).cuda()
    # support_features[0, 0] = 1
    # support_features[1, 1] = 1
    # query_features = torch.linspace(1,24,24).reshape(2, 3, 2, 2).cuda()
    # print_weights(support_features)
    # print_weights(query_features)
    support_features = support_features.permute(1, 0).unsqueeze(dim=2).repeat(1, 1, H * W)  # N x C x (H*W)
    support_features = support_features.unsqueeze(dim=0).repeat(B2, 1, 1, 1)  # B2 x N x C x (H*W)
    query_features = query_features.reshape(B2, C, H * W)  # B2 x C x (H*W)
    query_features = query_features.unsqueeze(dim=1).repeat(1, N_categories, 1, 1)  # B2 x N x C x (H*W)
    query_features = query_features * support_features  # B2 x N x C x (H*W)
    # modulated attentive features, B2 x N x C x H x W
    query_features = query_features.reshape(B2, N_categories, C, H, W)

    # print('=====4====')
    # print(support_features)
    # print(query_features)

    # print_weights(query_features)

    if compute_similarity == True:
        # note that here the query_features are attentive features, normed_similarity_map are l2 normed similarity map
        normed_similarity_map = compute_similarity_map(support_features_temp, query_features_temp, query_features)
    else:
        normed_similarity_map = None

    if (fused_attention is not None) or (output_attention_maps == True):
        # it should be copied in avoid to modify query_features
        # method 1: mean --> softmax
        # attention_maps = query_features.mean(dim=2).reshape(B2, N_categories, H*W)
        # method 2: max  --> softmax
        # attention_maps = query_features.max(dim=2)[0].reshape(B2, N_categories, H * W)
        # method 3: sum --> softmax
        attention_maps = query_features.sum(dim=2).reshape(B2, N_categories, H * W)  # note this is collapsed attentive features
        # method 4: sum --> relu --> softmax
        # attention_maps = nn.functional.relu(query_features.sum(dim=2)).reshape(B2, N_categories, H * W)
        # attention_maps = nn.functional.softmax(attention_maps, dim=2).reshape(B2, N_categories, H, W)
        # attention_maps_np = attention_maps.cpu().detach().numpy().copy()


    if fused_attention == 'L2-a':
        # l2 norm, spatial attention, applied to query features
        query_features_l2 = query_features_temp * query_features_temp
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W)
        attention_maps_l2 = attention_maps / query_features_l2.view(B2, 1, H * W)
        # softmax
        # attention_maps_l2 = nn.functional.softmax(attention_maps_l2, dim=2)     # B2 x N x (H*W)
        attention_maps_l2 = attention_maps_l2.reshape(B2, N_categories, H, W)   # B2 x N x H x W

        # fuse space-channel attention
        # attention_maps_l2 = attention_maps_l2.detach()

        # here the query feature is attentive feature
        query_features *= attention_maps_l2.view(B2, N_categories, 1, H, W)
    elif fused_attention == 'L2-b':
        # l2 norm, fuse spatial and channel attention, applied to query features
        query_features_l2 = query_features_temp * query_features_temp  # B x C x H x W
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W)  # B2 x (H*W)

        support_features_l2 = support_features_temp * support_features_temp  # C x N
        support_features_l2 = torch.sqrt(support_features_l2.sum(dim=0))  # N

        # norm = torch.matmul(query_features_l2.view(B2, H*W, 1), support_features_l2.view(1, N_categories)).permute(0, 2, 1)  # B2 x N x (H*W)
        norm = query_features_l2.unsqueeze(dim=1).repeat(1, N_categories, 1) * support_features_l2.view(1, N_categories, 1)  # B2 x N x (H*W)

        # here the query feature is attentive feature
        query_features /= (norm.reshape(B2, N_categories, 1, H, W) + 1e-9)   # B2 x N x C x H x W
    elif fused_attention == 'L2-c':
        # l2 norm, using spatial attentiom map directly
        query_features_l2 = query_features_temp * query_features_temp  # B x C x H x W
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W).detach()  # B2 x (H*W)

        # support_features_l2 = support_features_temp * support_features_temp  # C x N
        # support_features_l2 = torch.sqrt(support_features_l2.sum(dim=0)).detach()  # N

        # norm = torch.matmul(query_features_l2.view(B2, H*W, 1), support_features_l2.view(1, N_categories)).permute(0, 2, 1).detach()  # B2 x N x (H*W)
        # norm = query_features_l2.unsqueeze(dim=1).repeat(1, N_categories, 1) * support_features_l2.view(1, N_categories, 1)  # B2 x N x (H*W)

        # query_features = (attention_maps / (norm + 1e-9)).reshape(B2, N_categories, 1, H, W)   # B2 x N x 1 x H x W
        # query_features = (attention_maps / (query_features_l2 + 1e-9).view(B2, 1, H*W)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
        # query_features = (attention_maps / (support_features_l2 + 1e-9).unsqueeze(dim=0).repeat(B2, 1).view(B2, N_categories, 1)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W

        query_features = (attention_maps / (query_features_l2 + 1e-9).view(B2, 1, H*W)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
        # query_features = (attention_maps / (support_features_l2 + 1e-9).unsqueeze(dim=0).repeat(B2, 1).view(B2, N_categories, 1)).reshape(B2, N_categories, 1, H, W)  # B2 x N x 1 x H x W
    elif fused_attention == 'L1-a':
        # l1 norm, spatial attention, applied to query features
        query_features_l1 = torch.abs(query_features_temp)
        query_features_l1 = query_features_l1.sum(dim=1).reshape(B2, H * W)
        attention_maps_l1 = attention_maps / query_features_l1.view(B2, 1, H * W)
        # softmax
        # attention_maps_l1 = nn.functional.softmax(attention_maps_l1, dim=2)    # B2 x N x (H*W)
        attention_maps_l1 = attention_maps_l1.reshape(B2, N_categories, H, W)  # B2 x N x H x W

        # fuse space-channel attention
        # attention_maps_l1 = attention_maps_l1.detach()
        # here the query feature is attentive feature
        query_features *= attention_maps_l1.view(B2, N_categories, 1, H, W)
    elif fused_attention == 'attmap':
        # use attention map directly, average by number of feature channel
        query_features = (attention_maps / C).reshape(B2, N_categories, 1, H, W)
    else:  # None
        pass

    if output_attention_maps == True:
        # l2 norm
        query_features_l2 = query_features_temp * query_features_temp
        query_features_l2 = torch.sqrt(query_features_l2.sum(dim=1)).reshape(B2, H * W)
        attention_maps_l2 = attention_maps / query_features_l2.view(B2, 1, H * W)

        # l2 norm but using support fibers
        # support_features_l2 = support_features_temp * support_features_temp  # C x N
        # support_features_l2 = torch.sqrt(support_features_l2.sum(dim=0))  # N
        # attention_maps_l2 = (attention_maps / support_features_l2.unsqueeze(dim=0).repeat(B2, 1).view(B2, N_categories, 1)).reshape(B2, N_categories, 1, H, W)  # B2 x N x H x W

        # softmax
        attention_maps_l2 = nn.functional.softmax(attention_maps_l2, dim=2)    # B2 x N x (H*W)
        attention_maps_l2 = attention_maps_l2.reshape(B2, N_categories, H, W)  # B2 x N x H x W

        # l1 norm
        query_features_l1 = torch.abs(query_features_temp)
        query_features_l1 = query_features_l1.sum(dim=1).reshape(B2, H * W)
        attention_maps_l1 = attention_maps / query_features_l1.view(B2, 1, H * W)
        # softmax
        attention_maps_l1 = nn.functional.softmax(attention_maps_l1, dim=2)    # B2 x N x (H*W)
        attention_maps_l1 = attention_maps_l1.reshape(B2, N_categories, H, W)  # B2 x N x H x W

        # copy attention maps, tensor, B2 x N x H x W
        attention_maps_l2_cpu = copy.deepcopy(attention_maps_l2.cpu().detach())
        attention_maps_l1_cpu = copy.deepcopy(attention_maps_l1.cpu().detach())
    else:
        attention_maps_l2_cpu, attention_maps_l1_cpu = None, None


    return query_features, attention_maps_l2_cpu, attention_maps_l1_cpu, normed_similarity_map

class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, conv1_stride=1, conv2_stride=1, downsample_using_conv=True):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 256, 3, conv1_stride, 1)
        self.conv2 = nn.Conv2d(256, output_channel, 1, conv2_stride, 0)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.downsample_using_conv = downsample_using_conv
        downsample_rate = conv1_stride * conv2_stride
        if self.downsample_using_conv == True:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=downsample_rate, padding=0),
                # nn.BatchNorm2d(256)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample_using_conv == True:
            identity = self.skip_connection(identity)
            out += identity
        else:  # direct addition when self.downsample_using_conv=False, at this time, the input_channel = output_channel = 256 and map size does not change
            out += identity

        out = self.relu(out)
        out = self.maxpool(out)

        return out

class HourglassBackbone(nn.Module):
    def __init__(self, input_channel=2048):
        super(HourglassBackbone, self).__init__()

        self.cls_up1 = nn.Sequential(
            nn.Conv2d(input_channel, 512, kernel_size=3, stride=2, padding=0),
        )
        self.cls_up2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
        )
        self.cls_up3 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=0),
        )
        self.cls_low3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=0),

        )
        self.cls_low2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        # transform (B * N) x C x H x W features
        (B_N, C, H, W) = x.shape

        # B x N x C x H x W --> (B*N) x C x H x W
        # out = x.reshape(B * N, C, H, W)

        out_up1 = self.cls_up1(x)
        out_up2 = self.cls_up2(out_up1)
        out_up3 = self.cls_up3(out_up2)
        out_low3 = self.cls_low3(out_up3)
        out_low3_2 = out_low3 + out_up2
        out_low2 = self.cls_low2(out_low3_2)
        out_low2_2 = out_low2 + out_up1

        return out_low2_2  # (B*N) x 512 x 5 x 5 if x is (B*N) x C x 12 x 12


class DescriptorNet(nn.Module):
    def __init__(self, input_channel = 512, conv_blocks = 4, output_fiber=True, specify_fc_out_units = None, architecture_type = 'type1', net_config=(256, 1024)):
        super(DescriptorNet, self).__init__()

        self.architecture_type = architecture_type
        self.output_fiber = output_fiber

        if self.architecture_type == 'type1':
            self.convs = nn.Sequential()

            convs_out_channel = 512  # needs to be tuned manually
            channel_intervals = [input_channel]
            medium_channel = int(math.sqrt(input_channel * convs_out_channel))
            for i in range(conv_blocks-1):
                channel_intervals.append(medium_channel)
            channel_intervals.append(convs_out_channel)
            for i in range(conv_blocks):
                self.convs.add_module('conv_{}'.format(i), nn.Conv2d(channel_intervals[i], channel_intervals[i+1], 5, 1, 2))
                self.convs.add_module('relu_{}'.format(i), nn.ReLU(True))
                self.convs.add_module('pool_{}'.format(i), nn.MaxPool2d(3, 2, 1))

        elif self.architecture_type == 'type2':
            self.convs = nn.Sequential()
            convs_out_channel = 512  # needs to be tuned manually
            self.convs.add_module('residual_block_start', ResidualBlock(input_channel, 256, conv1_stride=2, conv2_stride=1, downsample_using_conv=True))
            if conv_blocks > 2:
                for i in range(conv_blocks - 2):
                    self.convs.add_module('residual_block%s'%i, ResidualBlock(256, 256, downsample_using_conv=False))
            self.convs.add_module('residual_block_end', ResidualBlock(256, convs_out_channel, conv1_stride=1, conv2_stride=1, downsample_using_conv=True))

        elif self.architecture_type == 'type3':
            convs_out_channel = net_config[1] # 1024  # outmap: 1024 * 2 * 2
            mediate_channel = net_config[0] # 1024, 256
            self.convs = nn.Sequential()
            self.convs.add_module('block0_conv', nn.Conv2d(input_channel, mediate_channel, kernel_size=3, stride=2, padding=1)),
            self.convs.add_module('block0_relu', nn.ReLU(True))
            for i in range(conv_blocks - 2):
                self.convs.add_module('block%d_conv'%(i + 1), nn.Conv2d(mediate_channel, mediate_channel, kernel_size=3, stride=2, padding=1)),
                self.convs.add_module('block%d_relu'%(i + 1), nn.ReLU(True))
            self.convs.add_module('block%d_conv'%(conv_blocks-1), nn.Conv2d(mediate_channel, convs_out_channel, kernel_size=3, stride=2, padding=1)),
            self.convs.add_module('block%d_relu'%(conv_blocks-1), nn.ReLU(True))

        elif self.architecture_type == 'type3-2':
            convs_out_channel = net_config[1] # 1024  # outmap: 1024 * 2 * 2
            mediate_channel = net_config[0] # 256, 1024
            self.convs = nn.Sequential()
            self.convs.add_module('block0_conv', nn.Conv2d(input_channel, mediate_channel, kernel_size=1, stride=1, padding=0)),  # 1x1 conv
            self.convs.add_module('block0_relu', nn.ReLU(True))
            for i in range(conv_blocks - 1):
                self.convs.add_module('block%d_conv'%(i + 1), nn.Conv2d(mediate_channel, mediate_channel, kernel_size=3, stride=2, padding=1)),
                self.convs.add_module('block%d_relu'%(i + 1), nn.ReLU(True))
            self.convs.add_module('block%d_conv'%(conv_blocks-1), nn.Conv2d(mediate_channel, convs_out_channel, kernel_size=3, stride=2, padding=1)),
            self.convs.add_module('block%d_relu'%(conv_blocks-1), nn.ReLU(True))

        elif self.architecture_type == 'type4':
            self.convs = HourglassBackbone(input_channel=input_channel)

        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    m.bias.data.zero_()

        self.specify_fc_out_units = specify_fc_out_units
        if specify_fc_out_units != None:
            linear_in_channel = convs_out_channel * 2 * 2  # needs to be tuned manually
            self.fc = nn.Sequential(
                nn.Linear(linear_in_channel, specify_fc_out_units),
                # nn.BatchNorm1d(linear_out_channel),
                nn.ReLU(True)
            )

            # nn.init.normal_(self.convs[0].weight, 0, 0.01)
            self.fc[0].weight.data.normal_(mean=0, std=0.01)
            self.fc[0].bias.data.zero_()

        print('DescriptorNet initialized!')

    def forward(self, x: torch.Tensor):
        # transform B x N x C x H x W attentive features to B x N x D keypoint descriptors
        (B, N, C, H, W) = x.shape
        out = x.reshape(B*N, C, H, W)
        # (B*N) x C x H x W --> (B*N) x C' x H' x W'
        out = self.convs(out)

        if self.output_fiber == True:
            if self.specify_fc_out_units:
                # (B*N) x C' x H' x W'  --> (B*N) x (C' * H' * W') --> (B*N) x D
                out = out.reshape(B*N, -1)
                out = self.fc(out)

            out = out.reshape(B, N, -1)  # B x N x D, where D is the desired linear_out_channel
        else:
            (c, h, w) = out.shape[1:]
            out = out.reshape(B, N, c, h, w)  # output a feature map

        return out


class RegressorVanila(nn.Module):
    def __init__(self, input_channel = 512):
        '''
        Vanila version regressor will regress N keypoint descriptors to N keypoints

        :param x:
        :param input_channel:
        :param N_categories:
        '''
        super(RegressorVanila, self).__init__()
        self.fc1 = nn.Sequential(
            # nn.Linear(input_channel*N_categories, 1024),
            nn.Linear(input_channel, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(True)
        )
        self.fc3 = nn.Sequential(
            # nn.Linear(512, N_categories*2),
            nn.Linear(512, 2),
            # nn.Sigmoid()  # regress the keypoint location from 0~1
        )

        nn.init.normal_(self.fc1[0].weight, mean=0, std=0.01)
        nn.init.normal_(self.fc2[0].weight, mean=0, std=0.01)
        nn.init.normal_(self.fc3[0].weight, mean=0, std=0.01)


        print('RegressorVanila initialized!')

    def forward(self, x):
        need_reshape = False
        out = x
        if len(x.size()) == 3:
            need_reshape = True
            B, N, D = x.size()
            out = x.reshape(B*N, D)

        # transform B x N x D keypoint descriptors to B x N x 2 location predictions
        out = self.fc1(out)  # B x N x D
        out = self.fc2(out)
        out = self.fc3(out)  # B x N x 2

        if need_reshape == True:
            out = out.reshape(B, N, -1)

        return out



# this CNNModel is used for handwritten digit recognition, 10 classes
class ClassifierGRL(nn.Module):
    def __init__(self, input_channel=1024, output_classes=4, architecture_type='type3'):
        super(ClassifierGRL, self).__init__()

        self.architecture_type = architecture_type

        if self.architecture_type == 'type1':
            # # Architecture 1
            self.convs = nn.Sequential(
                nn.Conv2d(input_channel, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                # nn.AdaptiveAvgPool2d(output_size=(6, 6))
            )

            self.fc = nn.Sequential(
                # nn.Linear( 256 * 4 * 4, 1024),  # need to adjust mannualy
                # nn.BatchNorm1d(1024),
                # nn.ReLU(inplace=True),
                # nn.Linear(1024, 512),
                # nn.BatchNorm1d(512),
                # nn.ReLU(inplace=True),
                nn.Linear(256 * 6 * 6, output_classes),  # need to adjust mannualy
                nn.LogSoftmax(dim=1)
            )
        elif self.architecture_type == 'type2':
            # Architecture 2
            self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(512)
            self.relu1 = nn.ReLU(inplace=True)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(256)
            self.relu2 = nn.ReLU(inplace=True)
            self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.downsample = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(256)
            )

            self.fc = nn.Sequential(
                nn.Linear(256 * 6 * 6, output_classes),   # need to adjust mannualy
                nn.LogSoftmax(dim=1)
            )
        elif self.architecture_type == 'type3':
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            # self.pool = nn.AdaptiveMaxPool2d((2, 2))
            self.fc = nn.Sequential(
                nn.Linear(input_channel * 2 * 2, output_classes),
                nn.LogSoftmax(dim=1)
            )
        elif self.architecture_type == 'type3-2':
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            # self.pool = nn.AdaptiveMaxPool2d((2, 2))
            self.fc = nn.Sequential(
                nn.Linear(input_channel * 2 * 2, 512),
                # nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, output_classes),
                nn.LogSoftmax(dim=1)
            )
        elif self.architecture_type == 'type4':
            # self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.pool = nn.AdaptiveMaxPool2d((2, 2))
            self.fc = nn.Sequential(
                nn.Linear(input_channel * 2 * 2, output_classes),
                nn.LogSoftmax(dim=1)
            )
        elif self.architecture_type == 'type4-2':
            # self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.pool = nn.AdaptiveMaxPool2d((2, 2))
            self.fc = nn.Sequential(
                nn.Linear(input_channel * 2 * 2, 512),
                # nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, output_classes),
                nn.LogSoftmax(dim=1)
            )

        # for m in self.convs.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(mean=0, std=0.01)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        # for m in self.linear_layers.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.normal_(mean=0, std=0.01)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x: torch.Tensor, alpha: float = 0):
        batch_size = x.size(0)

        out = ReverseLayerF.apply(x, alpha)  # reverse_feature

        if self.architecture_type == 'type1':
            # Architecture 1
            out = self.convs(out)
            out = self.fc(out.reshape(batch_size, -1))

        elif self.architecture_type == 'type2':
            # Architecture 2
            identity = out

            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.maxpool1(out)

            out = self.conv2(out)
            out = self.bn2(out)

            identity = self.downsample(identity)
            out += identity

            out = self.relu2(out)
            out = self.maxpool2(out)

            out = self.fc(out.reshape(batch_size, -1))

        elif self.architecture_type == 'type3' or self.architecture_type == 'type3-2':
            out = self.pool(out)
            out = self.fc(out.reshape(batch_size, -1))

        elif self.architecture_type == 'type4' or self.architecture_type == 'type4-2':
            out = self.pool(out)
            out = self.fc(out.reshape(batch_size, -1))

        return out


class GridBasedLocator(nn.Module):
    def __init__(self, grid_length=8, use_one_hot_code=True):
        super(GridBasedLocator, self).__init__()

        self.grid_length = grid_length
        self.use_one_hot_code = use_one_hot_code
        # self.convs = nn.Sequential()
        # self.convs.add_module('conv0', nn.Conv2d(input_channel, 1024, kernel_size=3, stride=2, padding=1)),
        # self.convs.add_module('relu0', nn.ReLU(True))
        # for i in range(conv_layers - 1):
        #     self.convs.add_module('conv%d'%(i + 1), nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)),
        #     self.convs.add_module('relu%d'%(i + 1), nn.ReLU(True))

        linear_in_channel = 1024 * 2 * 2  # needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            nn.LogSoftmax(dim=1)
        )

        if use_one_hot_code:
            self.linear_grid_deviation = nn.Sequential(
                nn.Linear(grid_length * grid_length + linear_in_channel, 2)
            )
        else:
            self.linear_grid_deviation = nn.Sequential(
                nn.Linear(linear_in_channel, 2)
            )

    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None):
        # transform B x N x D features
        (B, N, D) = x.shape

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class = self.linear_grid_class(out)  # (B*N) * (L*L)

        if self.use_one_hot_code:
            if training_phase == True:
                one_hot_grid_label = one_hot_grid_label.reshape(B*N, -1)
                out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
            else:  # testing
                # compute one-hot grid label for predicted grids
                predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
                one_hot_grid_label = torch.zeros(B * N, self.grid_length**2).cuda()
                one_hot_grid_label = one_hot_grid_label.scatter(dim=1, index=torch.unsqueeze(predict_grids, dim=1), value=1)  # (B2 * N) x (grid_length * grid_length)
                # temp = one_hot_grid_label.cpu().detach().numpy()
                # print(temp)
                out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
                # temp = out.cpu().detach().numpy()
                # print(temp)
        grid_deviation = self.linear_grid_deviation(out)  # (B * N) * 2

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 2

        return grid_class, grid_deviation

class GridBasedLocator1(nn.Module):
    def __init__(self, grid_length=8):
        super(GridBasedLocator1, self).__init__()

        self.grid_length = grid_length
        # self.convs = nn.Sequential()
        # self.convs.add_module('conv0', nn.Conv2d(input_channel, 1024, kernel_size=3, stride=2, padding=1)),
        # self.convs.add_module('relu0', nn.ReLU(True))
        # for i in range(conv_layers - 1):
        #     self.convs.add_module('conv%d'%(i + 1), nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)),
        #     self.convs.add_module('relu%d'%(i + 1), nn.ReLU(True))

        linear_in_channel = 1024 * 2 * 2  # needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            nn.LogSoftmax(dim=1)
        )


        self.linear_grid_deviation = nn.Sequential(
            nn.Linear(grid_length * grid_length + linear_in_channel, 4)
        )

    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None):
        # transform B x N x D features
        (B, N, D) = x.shape

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class = self.linear_grid_class(out)  # (B*N) * (L*L)

        if training_phase == True:
            one_hot_grid_label = one_hot_grid_label.reshape(B*N, -1)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
        else:  # testing
            # compute one-hot grid label for predicted grids
            predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
            one_hot_grid_label = torch.zeros(B * N, self.grid_length**2).cuda()
            one_hot_grid_label = one_hot_grid_label.scatter(dim=1, index=torch.unsqueeze(predict_grids, dim=1), value=1)  # (B2 * N) x (grid_length * grid_length)
            # temp = one_hot_grid_label.cpu().detach().numpy()
            # print(temp)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
            # temp = out.cpu().detach().numpy()
            # print(temp)

        deviation_part = self.linear_grid_deviation(out)  # (B * N) * 4

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        deviation_part = deviation_part.reshape(B, N, -1)  # B x N x 4  (tx, ty, logvar_x, logvar_y)
        grid_deviation, logvar = torch.split(deviation_part, (2,2), dim=2)

        return grid_class, grid_deviation, logvar


class GridBasedLocator2(nn.Module):
    def __init__(self, grid_length=8):
        super(GridBasedLocator2, self).__init__()

        self.grid_length = grid_length

        linear_in_channel = 1024 * 2 * 2  # needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            nn.LogSoftmax(dim=1)
        )

        self.linear_grid_deviation = nn.Sequential(
            nn.Linear(grid_length * grid_length + linear_in_channel, 2)
        )

        self.linear_dev_uncertainty = nn.Sequential(
            nn.Linear(grid_length * grid_length + linear_in_channel, 2)
        )

    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None):
        # transform B x N x D features
        (B, N, D) = x.shape

        # B x N x D --> (B*N) x D
        out = x.reshape(B * N, -1)

        grid_class = self.linear_grid_class(out)  # (B*N) * (L*L)

        if training_phase == True:
            one_hot_grid_label = one_hot_grid_label.reshape(B * N, -1)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
        else:  # testing
            # compute one-hot grid label for predicted grids
            predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
            one_hot_grid_label = torch.zeros(B * N, self.grid_length ** 2).cuda()
            one_hot_grid_label = one_hot_grid_label.scatter(dim=1, index=torch.unsqueeze(predict_grids, dim=1),
                                                            value=1)  # (B2 * N) x (grid_length * grid_length)
            # temp = one_hot_grid_label.cpu().detach().numpy()
            # print(temp)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
            # temp = out.cpu().detach().numpy()
            # print(temp)

        grid_deviation = self.linear_grid_deviation(out)  # (B * N) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.
        logvar = self.linear_dev_uncertainty(out)

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 4  (tx, ty, logvar_x, logvar_y)
        logvar = logvar.reshape(B, N, -1)

        return grid_class, grid_deviation, logvar


def sampler(mean, sigma, sample_times=1, set_negative_noise=False):
    size = mean.shape
    if sample_times == 1:
        noise = torch.randn(size)
        if set_negative_noise == True:
            noise = -torch.abs(noise)
        if torch.cuda.is_available():
            noise = noise.cuda()
        value = mean + sigma * noise
    else:
        size = (*size, sample_times)
        ndims = len(size)
        noise = torch.randn(size)
        if set_negative_noise == True:
            noise = -torch.abs(noise)
        if torch.cuda.is_available():
            noise = noise.cuda()
        value = mean.unsqueeze(dim=ndims-1) + sigma.unsqueeze(dim=ndims-1) * noise  # using broadcast
    return value

def compute_location_expectation(p, L, method=0):
    '''
    compute location expectation
    :param p: (L*L)
    :param L: length for grids
    :param method: the method how to compute the expection
            method 0, using all probs
            method 1, using max probs
            method 2, using a neighborhood centered at the location with max prob
    :return: location expectation (x, y)
    '''
    grid_range = torch.linspace(0, L - 1, L)
    # print(grid_range)
    xx, yy = torch.meshgrid(grid_range, grid_range)
    # print(xx)
    # print(yy)
    xx, yy = xx.unsqueeze(dim=2), yy.unsqueeze(dim=2)
    grids = torch.cat((xx, yy), dim=2).reshape(-1, 2)  # (L*L) * 2, each element (y, x) is like (0, 0), (0, 1), .., (L-1, L-1)
    # print(grids)
    # l = torch.linspace(0, L ** 2 - 1, L ** 2)
    # print(l.reshape(L, L))
    if method == 0:  # naive mean
        p = p.view(L * L, 1)
        expectation = torch.sum( p*grids, dim=0)  # using broadcast
        # ind = torch.max(p, dim=0)[1]
        # print(ind // L, ind%L)
    elif method == 1:  # using location where has max probability
        ind = torch.max(p, dim=0)[1]
        expectation = torch.FloatTensor([ind // L, ind % L])  # (gridy, gridx)
    elif method == 2:  # using the index set of neighborhood centered at a grid cell with max probability
        window = 3  # should be odd number
        half_window = window // 2
        ind = torch.max(p, dim=0)[1]  # compute the index with max probability
        cx, cy = ind % L, ind // L    # compute center
        ind_set = []  # compute the index set of points in the neighborhood
        for iy in range(-half_window, half_window+1, 1):
            for ix in range(-half_window, half_window+1, 1):
                new_x = cx + ix
                new_y = cy + iy
                if new_y < 0 or new_y >= L or new_x < 0 or new_y >= L:
                    continue
                new_ind = new_y * L + new_x
                ind_set.append(new_ind)
        K = len(ind_set)  # K quality points in the neighborhood
        expectation = torch.zeros(2)  # (y, x)
        local_p_sum = torch.tensor(0.)
        for i in range(K):
            # ratio = p[ind_set[i]] / local_p_sum
            p_i = p[ind_set[i]]
            expectation[0] += p_i * (ind_set[i] // L)  # y
            expectation[1] += p_i * (ind_set[i] % L)   # x
            local_p_sum += p_i
        expectation /= local_p_sum

    # exchange y, x coordinates
    expectation = torch.tensor([expectation[1], expectation[0]])  # (x, y)
    return expectation

def compute_location_var(z, L, T, method=0):
    '''
    compute the variance for predicted location according to T times of sampling; the grid scale is L x L
    :param z: (B*N) * (L*L) * T, sampled activations
    :param L: grid length
    :param T: T times of sampling
    :param method: the method that specifies how to compute expectation for location using output probability distribution p(g|z(u, sigma))
    :return: (B*N) * 2, each row is location variance (var(x), var(y))
    '''
    p = z.softmax(dim=1)    # (B*N) * (L*L) * T
    mean_p = p.mean(dim=2)  # (B*N) * (L*L)
    num = p.shape[0]        # (B*N)
    estimate_locations = torch.zeros(num, 2, T)  # (B*N) * 2 * T
    mean_location = torch.zeros(num, 2)          # (B*N) * 2
    for i in range(num):
        for t in range(T):
            location = compute_location_expectation(p[i, :, t], L, method)
            estimate_locations[i, :, t] = location
    for i in range(num):
        location = compute_location_expectation(mean_p[i, :], L, method)
        mean_location[i, :] = location
    # var = (estimate_locations ** 2).mean(dim=2) - mean_location**2  # (B*N) * 2
    var = ((estimate_locations - mean_location.view(num, 2, 1)) ** 2).mean(dim=2)

    return var

class GridBasedLocator3(nn.Module):
    def __init__(self, grid_length=8, sample_times=15):
        super(GridBasedLocator3, self).__init__()

        self.grid_length = grid_length
        self.sample_times = sample_times

        linear_in_channel = 1024 * 2 * 2  # needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            # nn.LogSoftmax(dim=1)
        )

        self.linear_class_uncertainty = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
        )

        self.linear_grid_deviation = nn.Sequential(
            nn.Linear(grid_length * grid_length + linear_in_channel, 2)
        )

        # self.linear_dev_uncertainty = nn.Sequential(
        #     nn.Linear(grid_length * grid_length + linear_in_channel, 2)
        # )

    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None):
        # transform B x N x D features
        (B, N, D) = x.shape

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class_u = self.linear_grid_class(out)  # (B*N) * (L*L)
        # grid_class_u_np = grid_class_u.cpu().detach().numpy()
        # grid_class_u_np2 = torch.log_softmax(grid_class_u, dim=1).cpu().detach().numpy()

        # apply heteroscedastic uncertainty over classification, using Monte Carlo simulation to
        # estimate the probability output
        T = self.sample_times  # 30
        rho = self.linear_class_uncertainty(out)  # (B*N) * (L*L)
        sigmas = torch.nn.functional.softplus(rho)# (B*N) * (L*L)
        # sigmas_np = sigmas.cpu().detach().numpy()
        sample_z = sampler(grid_class_u, sigmas, sample_times=T) # (B*N) * (L*L) * T
        # sample_z_np = sample_z.cpu().detach().numpy()
        grid_class_prob = sample_z.softmax(dim=1).mean(dim=2)  # Monte Carlo Simulation, (B*N) * (L*L)
        grid_class_prob_np = grid_class_prob.cpu().detach().numpy()
        grid_class = torch.log(grid_class_prob)
        # grid_class_np = grid_class.cpu().detach().numpy()


        if training_phase == True:
            one_hot_grid_label = one_hot_grid_label.reshape(B*N, -1)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
        else:  # testing
            # compute one-hot grid label for predicted grids
            predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
            one_hot_grid_label = torch.zeros(B * N, self.grid_length**2).cuda()
            one_hot_grid_label = one_hot_grid_label.scatter(dim=1, index=torch.unsqueeze(predict_grids, dim=1), value=1)  # (B2 * N) x (grid_length * grid_length)
            # temp = one_hot_grid_label.cpu().detach().numpy()
            # print(temp)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
            # temp = out.cpu().detach().numpy()
            # print(temp)
        grid_deviation = self.linear_grid_deviation(out)  # (B * N) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.
        # logvar = self.linear_dev_uncertainty(out)

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 4  (tx, ty, logvar_x, logvar_y)
        # logvar = logvar.reshape(B, N, -1)
        # sigmas_np2 = torch.exp(logvar).cpu().detach().numpy()

        return grid_class, grid_deviation, None  # logvar


class GridBasedLocator4(nn.Module):
    def __init__(self, grid_length=8, sample_times=15):
        super(GridBasedLocator4, self).__init__()

        self.grid_length = grid_length
        self.sample_times = sample_times

        linear_in_channel = 1024 * 2 * 2  # needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            # nn.LogSoftmax(dim=1)
        )

        self.linear_class_uncertainty = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
        )

        self.linear_grid_deviation = nn.Sequential(
            nn.Linear(grid_length * grid_length + linear_in_channel, 2)
        )

        self.linear_dev_uncertainty = nn.Sequential(
            nn.Linear(grid_length * grid_length + linear_in_channel, 2)
        )

    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None):
        # transform B x N x D features
        (B, N, D) = x.shape

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class_u = self.linear_grid_class(out)  # (B*N) * (L*L)
        # grid_class_u_np = grid_class_u.cpu().detach().numpy()
        # grid_class_u_np2 = torch.log_softmax(grid_class_u, dim=1).cpu().detach().numpy()

        # apply heteroscedastic uncertainty over classification, using Monte Carlo simulation to
        # estimate the probability output
        T = self.sample_times  # 30
        rho = self.linear_class_uncertainty(out)  # (B*N) * (L*L)
        sigmas = torch.nn.functional.softplus(rho)# (B*N) * (L*L)
        # sigmas_np = sigmas.cpu().detach().numpy()
        sample_z = sampler(grid_class_u, sigmas, sample_times=T) # (B*N) * (L*L) * T
        # sample_z_np = sample_z.cpu().detach().numpy()
        grid_class_prob = sample_z.softmax(dim=1).mean(dim=2)  # Monte Carlo Simulation, (B*N) * (L*L)
        grid_class_prob_np = grid_class_prob.cpu().detach().numpy()
        grid_class = torch.log(grid_class_prob)
        # grid_class_np = grid_class.cpu().detach().numpy()


        if training_phase == True:
            one_hot_grid_label = one_hot_grid_label.reshape(B*N, -1)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
        else:  # testing
            # compute one-hot grid label for predicted grids
            predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
            one_hot_grid_label = torch.zeros(B * N, self.grid_length**2).cuda()
            one_hot_grid_label = one_hot_grid_label.scatter(dim=1, index=torch.unsqueeze(predict_grids, dim=1), value=1)  # (B2 * N) x (grid_length * grid_length)
            # temp = one_hot_grid_label.cpu().detach().numpy()
            # print(temp)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
            # temp = out.cpu().detach().numpy()
            # print(temp)
        grid_deviation = self.linear_grid_deviation(out)  # (B * N) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.
        logvar = self.linear_dev_uncertainty(out)

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 4  (tx, ty, logvar_x, logvar_y)
        logvar = logvar.reshape(B, N, -1)
        # sigmas_np2 = torch.exp(logvar).cpu().detach().numpy()

        return grid_class, grid_deviation, logvar


class GridBasedLocatorX(nn.Module):
    def __init__(self, grid_length=8, reg_uncertainty=True, cls_uncertainty=True, covar_Q_size=None, sample_times=15, negative_noise=False, sigma_computing_mode=0, alpha=1, beta=0):
        super(GridBasedLocatorX, self).__init__()

        self.grid_length = grid_length
        self.sample_times = sample_times
        self.negative_noise = negative_noise
        self.reg_uncertainty = reg_uncertainty
        self.cls_uncertainty = cls_uncertainty
        self.sigma_computing_mode = sigma_computing_mode
        if sigma_computing_mode == 2:
            self.alpha = alpha
            self.beta  = beta

        linear_in_channel = 1024 * 2 * 2  # 1/32, 1/46, 1/64, needs to be tuned manually
        # linear_in_channel = 1024        # 1/46, 1/64, needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            # nn.LogSoftmax(dim=1)
        )

        if self.cls_uncertainty == True:
            self.linear_class_uncertainty = nn.Sequential(
                nn.Linear(linear_in_channel, grid_length * grid_length),
            )

        self.linear_grid_deviation = nn.Linear(grid_length * grid_length + linear_in_channel, 2)

        if self.reg_uncertainty == True:
            if covar_Q_size is None:
                self.linear_dev_uncertainty = nn.Linear(grid_length * grid_length + linear_in_channel, 2)

            else:
                self.linear_dev_uncertainty = nn.Linear(grid_length * grid_length + linear_in_channel, covar_Q_size[0]*covar_Q_size[1])


    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None, compute_grid_var=False):
        # transform B x N x D features
        (B, N, D) = x.shape

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class_u = self.linear_grid_class(out)  # (B*N) * (L*L)
        # grid_class_u_np = grid_class_u.cpu().detach().numpy()
        # grid_class_u_np2 = torch.log_softmax(grid_class_u, dim=1).cpu().detach().numpy()

        if self.cls_uncertainty == True:
            # apply heteroscedastic uncertainty over classification, using Monte Carlo simulation to
            # estimate the probability output
            T = self.sample_times  # 30
            rho = self.linear_class_uncertainty(out)  # (B*N) * (L*L)

            if self.sigma_computing_mode == 0:  # sigma = log(1+exp(x))
                sigmas = torch.nn.functional.softplus(rho)# (B*N) * (L*L)
            elif self.sigma_computing_mode == 1:  # x = log(sigma^2)
                sigmas = torch.exp(rho/2)
            elif self.sigma_computing_mode == 2:  # sigma = alpha * sigmoid(x) + beta
                sigmas = self.alpha * torch.sigmoid(rho) + self.beta
            elif self.sigma_computing_mode == 3:  # max_{a}(x, 0) = (x + sqrt(x^2 + a))/2
                sigmas = (rho + torch.sqrt(rho ** 2 + self.beta)) / 2

            # draw image for grid sigmas
            # sigmas_np = sigmas.cpu().detach().numpy()
            # sigma_ims = make_uncertainty_map(sigmas_np, B)  # B x 368 x 368
            # sigma_ims_in_one = make_grid_images(torch.Tensor(sigma_ims).reshape(B, 1, 368, 368), denormalize=False)
            # sigma_ims_in_one = sigma_ims_in_one[:, :, 0].numpy()
            # save_plot_image(sigma_ims_in_one, 'sigma.png', does_show=False)

            sample_z = sampler(grid_class_u, sigmas, sample_times=T, set_negative_noise=self.negative_noise) # (B*N) * (L*L) * T
            # sample_z_np = sample_z.cpu().detach().numpy()
            grid_class_prob = sample_z.softmax(dim=1).mean(dim=2)  # Monte Carlo Simulation, (B*N) * (L*L)
            # grid_class_prob_np = (grid_class_prob).cpu().detach().numpy()
            grid_class = torch.log(grid_class_prob + 1e-9)
            # grid_class_np = grid_class.cpu().detach().numpy()

            # compute Variance var(g)
            if compute_grid_var:
                grid_var = compute_location_var(sample_z.cpu().detach(), self.grid_length, T, method=2)  # (B*N) * 2
                grid_var = grid_var.reshape(B, N, 2)

        else:
            grid_class = torch.log_softmax(grid_class_u, dim=1)
            if compute_grid_var:  # no grid var for such situation
                grid_var = None


        if training_phase == True:
            one_hot_grid_label = one_hot_grid_label.reshape(B*N, -1)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
        else:  # testing
            # compute one-hot grid label for predicted grids
            predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
            one_hot_grid_label = torch.zeros(B * N, self.grid_length**2).cuda()
            one_hot_grid_label = one_hot_grid_label.scatter(dim=1, index=torch.unsqueeze(predict_grids, dim=1), value=1)  # (B2 * N) x (grid_length * grid_length)
            # temp = one_hot_grid_label.cpu().detach().numpy()
            # print(temp)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
            # temp = out.cpu().detach().numpy()
            # print(temp)
        grid_deviation = self.linear_grid_deviation(out)  # (B * N) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.

        if self.reg_uncertainty == True:
            rho_reg = self.linear_dev_uncertainty(out)
            rho_reg = rho_reg.reshape(B, N, -1)  # B x N x 2  or B x N x (covar_Q_size[0] * covar_Q_size[1])
            # sigmas_np2 = torch.exp(rho_reg).cpu().detach().numpy()

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 2  (tx, ty)

        if self.reg_uncertainty == True and compute_grid_var == False:
            return grid_class, grid_deviation, rho_reg, None
        elif self.reg_uncertainty == True and compute_grid_var == True:
            return grid_class, grid_deviation, rho_reg, grid_var
        else:
            return grid_class, grid_deviation, None, None


class GridBasedLocatorXCovar(nn.Module):
    def __init__(self, grid_length=8, reg_uncertainty=True, cls_uncertainty=True, ureg_type=0, covar_Q_size=None, sample_times=15, negative_noise=False, sigma_computing_mode=0, alpha=1, beta=0):
        super(GridBasedLocatorXCovar, self).__init__()

        self.grid_length = grid_length
        self.sample_times = sample_times
        self.negative_noise = negative_noise
        self.reg_uncertainty = reg_uncertainty
        self.cls_uncertainty = cls_uncertainty
        self.sigma_computing_mode = sigma_computing_mode
        if sigma_computing_mode == 2:
            self.alpha = alpha
            self.beta  = beta

        linear_in_channel = 1024 * 2 * 2  # 1/32, 1/46, 1/64, needs to be tuned manually
        # linear_in_channel = 1024        # 1/46, 1/64, needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            # nn.LogSoftmax(dim=1)
        )

        if self.cls_uncertainty == True:
            self.linear_class_uncertainty = nn.Sequential(
                nn.Linear(linear_in_channel, grid_length * grid_length),
            )

        self.linear_grid_deviation = nn.Linear(grid_length * grid_length + linear_in_channel, 2)

        if self.reg_uncertainty == True:
            self.ureg_type = ureg_type
            if ureg_type == 0:  # independent var for each point, B x N x 2
                self.linear_dev_uncertainty = nn.Linear(grid_length * grid_length + linear_in_channel, 2)

            elif ureg_type == 1: # covariance for each point, B x N x (2d)
                self.linear_dev_uncertainty = \
                    nn.Linear(grid_length * grid_length + linear_in_channel, covar_Q_size[0] * covar_Q_size[1])

            # =======================
            elif ureg_type == 2: # covariance for N points, B x (N x N x 2d), directly use descriptors
                self.linear_dev_uncertainty = \
                    nn.Linear(linear_in_channel * covar_Q_size[0], covar_Q_size[0] * covar_Q_size[1] * covar_Q_size[2] * covar_Q_size[3])

            elif ureg_type == 3: # covariance for N points, B x (N x N x 2d), use descriptors concat with one-hot code
                self.linear_dev_uncertainty = \
                    nn.Linear((grid_length * grid_length + linear_in_channel)*covar_Q_size[0], covar_Q_size[0] * covar_Q_size[1] * covar_Q_size[2] * covar_Q_size[3])
            # =======================


    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None, compute_grid_var=False):
        # transform B x N x D features
        (B, N, D) = x.shape

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class_u = self.linear_grid_class(out)  # (B*N) * (L*L)
        # grid_class_u_np = grid_class_u.cpu().detach().numpy()
        # grid_class_u_np2 = torch.log_softmax(grid_class_u, dim=1).cpu().detach().numpy()

        if self.cls_uncertainty == True:
            # apply heteroscedastic uncertainty over classification, using Monte Carlo simulation to
            # estimate the probability output
            T = self.sample_times  # 30
            rho = self.linear_class_uncertainty(out)  # (B*N) * (L*L)

            if self.sigma_computing_mode == 0:  # sigma = log(1+exp(x))
                sigmas = torch.nn.functional.softplus(rho)# (B*N) * (L*L)
            elif self.sigma_computing_mode == 1:  # x = log(sigma^2)
                sigmas = torch.exp(rho/2)
            elif self.sigma_computing_mode == 2:  # sigma = alpha * sigmoid(x) + beta
                sigmas = self.alpha * torch.sigmoid(rho) + self.beta
            elif self.sigma_computing_mode == 3:  # max_{a}(x, 0) = (x + sqrt(x^2 + a))/2
                sigmas = (rho + torch.sqrt(rho ** 2 + self.beta)) / 2

            # draw image for grid sigmas
            # sigmas_np = sigmas.cpu().detach().numpy()
            # sigma_ims = make_uncertainty_map(sigmas_np, B)  # B x 368 x 368
            # sigma_ims_in_one = make_grid_images(torch.Tensor(sigma_ims).reshape(B, 1, 368, 368), denormalize=False)
            # sigma_ims_in_one = sigma_ims_in_one[:, :, 0].numpy()
            # save_plot_image(sigma_ims_in_one, 'sigma.png', does_show=False)

            sample_z = sampler(grid_class_u, sigmas, sample_times=T, set_negative_noise=self.negative_noise) # (B*N) * (L*L) * T
            # sample_z_np = sample_z.cpu().detach().numpy()
            grid_class_prob = sample_z.softmax(dim=1).mean(dim=2)  # Monte Carlo Simulation, (B*N) * (L*L)
            # grid_class_prob_np = (grid_class_prob).cpu().detach().numpy()
            grid_class = torch.log(grid_class_prob + 1e-9)
            # grid_class_np = grid_class.cpu().detach().numpy()

            # compute Variance var(g)
            if compute_grid_var:
                grid_var = compute_location_var(sample_z.cpu().detach(), self.grid_length, T, method=2)  # (B*N) * 2
                grid_var = grid_var.reshape(B, N, 2)

        else:
            grid_class = torch.log_softmax(grid_class_u, dim=1)
            if compute_grid_var:  # no grid var for such situation
                grid_var = None

        if training_phase == True:
            one_hot_grid_label = one_hot_grid_label.reshape(B*N, -1)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
        else:  # testing
            # compute one-hot grid label for predicted grids
            predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
            one_hot_grid_label = torch.zeros(B * N, self.grid_length**2).cuda()
            one_hot_grid_label = one_hot_grid_label.scatter(dim=1, index=torch.unsqueeze(predict_grids, dim=1), value=1)  # (B2 * N) x (grid_length * grid_length)
            # temp = one_hot_grid_label.cpu().detach().numpy()
            # print(temp)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
            # temp = out.cpu().detach().numpy()
            # print(temp)
        grid_deviation = self.linear_grid_deviation(out)  # (B * N) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.

        if self.reg_uncertainty == True:
            # independent var for each point B x N x 2 or covariance for each point, B x N x (2d)
            if self.ureg_type == 0 or self.ureg_type == 1:
                rho_reg = self.linear_dev_uncertainty(out)
                rho_reg = rho_reg.reshape(B, N, -1)  # B x N x 2  or B x N x (covar_Q_size[0] * covar_Q_size[1])
                # sigmas_np2 = torch.exp(rho_reg).cpu().detach().numpy()
            # =======================
            # covariance for all keypoints, B x (N x N x 2d), input is descriptor
            elif self.ureg_type == 2:
                out = x.reshape(B, -1)  # B x (N*C)
                rho_reg = self.linear_dev_uncertainty(out)  # B x (N*N*2d)
                rho_reg = rho_reg.reshape(B, N, N, -1)  # B x N x N x (2d), each block is 2 x d

            # covariance for all keypoints, B x (N x N x 2d), input is descritpor concatenated with one-hot code
            elif self.ureg_type == 3:
                out = out.reshape(B, -1)  # B x (N*(C + L*L))
                rho_reg = self.linear_dev_uncertainty(out)  # B x (N*N*2d)
                rho_reg = rho_reg.reshape(B, N, N, -1)  # B x N x N x (2d), each block is 2 x d
            # =======================


        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 2  (tx, ty)

        if self.reg_uncertainty == True and compute_grid_var == False:
            return grid_class, grid_deviation, rho_reg, None
        elif self.reg_uncertainty == True and compute_grid_var == True:
            return grid_class, grid_deviation, rho_reg, grid_var
        else:
            return grid_class, grid_deviation, None, None


class GridBasedLocatorX2(nn.Module):
    def __init__(self, grid_length=8, reg_uncertainty=True, cls_uncertainty=True, covar_Q_size=None, sample_times=15, negative_noise=False, sigma_computing_mode=0, alpha=1, beta=0):
        super(GridBasedLocatorX2, self).__init__()

        self.grid_length = grid_length
        self.sample_times = sample_times
        self.negative_noise = negative_noise
        self.reg_uncertainty = reg_uncertainty
        self.cls_uncertainty = cls_uncertainty
        self.sigma_computing_mode = sigma_computing_mode
        if sigma_computing_mode == 2:
            self.alpha = alpha
            self.beta  = beta

        linear_in_channel = 1024 * 2 * 2  # 1/32, 1/46, 1/64, needs to be tuned manually
        # linear_in_channel = 1024        # 1/46, 1/64, needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            # nn.LogSoftmax(dim=1)
        )

        if self.cls_uncertainty == True:
            self.linear_class_uncertainty = nn.Sequential(
                nn.Linear(linear_in_channel, grid_length * grid_length),
            )

        self.linear_grid_deviation = nn.Linear(linear_in_channel, grid_length * grid_length * 2)  # L*L*2

        if self.reg_uncertainty == True:
            if covar_Q_size is None:
                self.linear_dev_uncertainty = nn.Linear(linear_in_channel, grid_length * grid_length * 2)  # L*L*2
            else:
                self.linear_dev_uncertainty = nn.Linear(linear_in_channel, grid_length * grid_length * covar_Q_size[0] * covar_Q_size[1])  # L*L*(2d)

    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None, compute_grid_var=False):
        # Note the one_hot_grid_label is not one hot code, it is the B x N query label, the value range is 0~L*L-1

        # transform B x N x D features
        (B, N, D) = x.shape
        L = self.grid_length

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class_u = self.linear_grid_class(out)  # (B*N) * (L*L)
        # grid_class_u_np = grid_class_u.cpu().detach().numpy()
        # grid_class_u_np2 = torch.log_softmax(grid_class_u, dim=1).cpu().detach().numpy()

        if self.cls_uncertainty == True:
            # apply heteroscedastic uncertainty over classification, using Monte Carlo simulation to
            # estimate the probability output
            T = self.sample_times  # 30
            rho = self.linear_class_uncertainty(out)  # (B*N) * (L*L)

            if self.sigma_computing_mode == 0:  # sigma = log(1+exp(x))
                sigmas = torch.nn.functional.softplus(rho)# (B*N) * (L*L)
            elif self.sigma_computing_mode == 1:  # x = log(sigma^2)
                sigmas = torch.exp(rho/2)
            elif self.sigma_computing_mode == 2:  # sigma = alpha * sigmoid(x) + beta
                sigmas = self.alpha * torch.sigmoid(rho) + self.beta
            elif self.sigma_computing_mode == 3:  # max_{a}(x, 0) = (x + sqrt(x^2 + a))/2
                sigmas = (rho + torch.sqrt(rho ** 2 + self.beta)) / 2

            # draw image for grid sigmas
            # sigmas_np = sigmas.cpu().detach().numpy()
            # sigma_ims = make_uncertainty_map(sigmas_np, B)  # B x 368 x 368
            # sigma_ims_in_one = make_grid_images(torch.Tensor(sigma_ims).reshape(B, 1, 368, 368), denormalize=False)
            # sigma_ims_in_one = sigma_ims_in_one[:, :, 0].numpy()
            # save_plot_image(sigma_ims_in_one, 'sigma.png', does_show=False)

            sample_z = sampler(grid_class_u, sigmas, sample_times=T, set_negative_noise=self.negative_noise) # (B*N) * (L*L) * T
            # sample_z_np = sample_z.cpu().detach().numpy()
            grid_class_prob = sample_z.softmax(dim=1).mean(dim=2)  # Monte Carlo Simulation, (B*N) * (L*L)
            # grid_class_prob_np = (grid_class_prob).cpu().detach().numpy()
            grid_class = torch.log(grid_class_prob + 1e-9)
            # grid_class_np = grid_class.cpu().detach().numpy()

            # compute Variance var(g)
            if compute_grid_var:
                grid_var = compute_location_var(sample_z.cpu().detach(), self.grid_length, T, method=2)  # (B*N) * 2
                grid_var = grid_var.reshape(B, N, 2)

        else:
            grid_class = torch.log_softmax(grid_class_u, dim=1)
            if compute_grid_var:  # no grid var for such situation
                grid_var = None

        grid_deviation_out = self.linear_grid_deviation(out)  # (B * N) * (L*L*2)
        # grid_deviation_out = grid_deviation_out.reshape(B * N, L*L, 2)  # (B * N) * (L*L) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.

        # if self.reg_uncertainty == True:
        #     rho_reg = self.linear_dev_uncertainty(out)  # (B * N) * (L*L*2) or (B * N) * (L*L*2d)
        #     # rho_reg = rho_reg.reshape(B * N, L*L, 2)

        # if training_phase == True:
        #     # kp_label = one_hot_grid_label.reshape(B * N, -1)            # (B*N) * 2
        #     # retrieval_index = kp_label[:, 1] * L + kp_label[:, 0]       # (B*N)
        #     retrieval_index = one_hot_grid_label.view(-1)   # (B*N)
        #     ind_temp = torch.linspace(0, B*N-1, B*N).long().view(-1).cuda()
        #     grid_deviation_out = grid_deviation_out.view(-1, 2)  # (B*N*L*L) * 2
        #     grid_deviation = grid_deviation_out[ind_temp * L* L + retrieval_index, :]  # (B*N) * 2
        #     if self.reg_uncertainty == True:
        #         rho_reg = rho_reg.view(B*N*L*L, -1)  # (B*N*L*L) * 2 or (B*N*L*L) * (covar_Q_size[0] * covar_Q_size[1])
        #         rho_reg = rho_reg[ind_temp * L * L + retrieval_index, :]  # (B*N) * 2  or (B*N) * (covar_Q_size[0] * covar_Q_size[1])
        #         rho_reg = rho_reg.reshape(B, N, -1)  # B x N x 2 Or B x N x (covar_Q_size[0] * covar_Q_size[1])
        # else:  # testing
        #     # compute one-hot grid label for predicted grids
        #     predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
        #     ind_temp = torch.linspace(0, B * N - 1, B * N).long().view(-1).cuda()
        #     grid_deviation_out = grid_deviation_out.view(-1, 2)  # (B*N*L*L) * 2
        #     grid_deviation = grid_deviation_out[ind_temp * L* L + predict_grids, :]  # (B*N) * 2
        #     if self.reg_uncertainty == True:
        #         rho_reg = rho_reg.view(B*N*L*L, -1)  # (B*N*L*L) * 2 or (B*N*L*L) * (covar_Q_size[0] * covar_Q_size[1])
        #         rho_reg = rho_reg[ind_temp * L * L + predict_grids, :]  # (B*N) * 2 or (B*N) * (covar_Q_size[0] * covar_Q_size[1])
        #         rho_reg = rho_reg.reshape(B, N, -1)  # B x N x 2 Or B x N x (covar_Q_size[0] * covar_Q_size[1])

        if training_phase == True:
            # kp_label = one_hot_grid_label.reshape(B * N, -1)            # (B*N) * 2
            # retrieval_index = kp_label[:, 1] * L + kp_label[:, 0]       # (B*N)
            retrieval_index = one_hot_grid_label.view(-1)   # (B*N)
        else:  # testing
            # compute one-hot grid label for predicted grids
            retrieval_index = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
        ind_temp = torch.linspace(0, B * N - 1, B * N).long().view(-1).cuda()
        grid_deviation_out = grid_deviation_out.view(-1, 2)  # (B*N*L*L) * 2
        grid_deviation = grid_deviation_out[ind_temp * L* L + retrieval_index, :]  # (B*N) * 2

        if self.reg_uncertainty == True:
            rho_reg = self.linear_dev_uncertainty(out)  # (B * N) * (L*L*2) or (B * N) * (L*L*2d)
            # rho_reg = rho_reg.reshape(B * N, L*L, 2)
            rho_reg = rho_reg.view(B*N*L*L, -1)  # (B*N*L*L) * 2 or (B*N*L*L) * (covar_Q_size[0] * covar_Q_size[1])
            rho_reg = rho_reg[ind_temp * L * L + retrieval_index, :]  # (B*N) * 2 or (B*N) * (covar_Q_size[0] * covar_Q_size[1])
            rho_reg = rho_reg.reshape(B, N, -1)  # B x N x 2 Or B x N x (covar_Q_size[0] * covar_Q_size[1])

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 2  (tx, ty)

        if self.reg_uncertainty == True and compute_grid_var == False:
            return grid_class, grid_deviation, rho_reg, None
        elif self.reg_uncertainty == True and compute_grid_var == True:
            return grid_class, grid_deviation, rho_reg, grid_var
        else:
            return grid_class, grid_deviation, None, None

class GridBasedLocatorX2Covar(nn.Module):
    def __init__(self, grid_length=8, reg_uncertainty=True, cls_uncertainty=True, ureg_type=0, covar_Q_size=None, sample_times=15, negative_noise=False, sigma_computing_mode=0, alpha=1, beta=0):
        super(GridBasedLocatorX2Covar, self).__init__()

        self.grid_length = grid_length
        self.sample_times = sample_times
        self.negative_noise = negative_noise
        self.reg_uncertainty = reg_uncertainty
        self.cls_uncertainty = cls_uncertainty
        self.sigma_computing_mode = sigma_computing_mode
        if sigma_computing_mode == 2:
            self.alpha = alpha
            self.beta  = beta

        linear_in_channel = 1024 * 2 * 2  # 1/32, 1/46, 1/64, needs to be tuned manually
        # linear_in_channel = 1024        # 1/46, 1/64, needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            # nn.LogSoftmax(dim=1)
        )

        if self.cls_uncertainty == True:
            self.linear_class_uncertainty = nn.Sequential(
                nn.Linear(linear_in_channel, grid_length * grid_length),
            )

        self.linear_grid_deviation = nn.Linear(linear_in_channel, grid_length * grid_length * 2)  # L*L*2

        if self.reg_uncertainty == True:
            self.ureg_type = ureg_type
            if ureg_type == 0:  # independent var for each point, stored in grids, B x N x L*L*2
                self.linear_dev_uncertainty = nn.Linear(linear_in_channel, grid_length * grid_length * 2) # L*L*2
            elif ureg_type == 1: # covariance for each point, stored in grids, B x N x [L*L*(2d)]
                self.linear_dev_uncertainty = nn.Linear(linear_in_channel, grid_length * grid_length * covar_Q_size[0] * covar_Q_size[1])  # L*L*(2d)
            # =======================
            elif ureg_type == 2: # covariance for all points, directly use descriptors, B x [N*N*2d]
                self.linear_dev_uncertainty = nn.Linear(linear_in_channel*covar_Q_size[0], covar_Q_size[0] * covar_Q_size[1] * covar_Q_size[2] * covar_Q_size[3])
            elif ureg_type == 3: # covariance for all points, use descriptors concat with one-hot code, B x [N*N*2d]
                self.linear_dev_uncertainty = nn.Linear((grid_length * grid_length + linear_in_channel)*covar_Q_size[0], covar_Q_size[0] * covar_Q_size[1] * covar_Q_size[2] * covar_Q_size[3])
            # elif ureg_type == 4: # covariance for all points, directly use descriptors, stored in grids, B x L x L x [N*N*2d]
            #     self.linear_dev_uncertainty = nn.Linear((grid_length * grid_length)*covar_Q_size[0], grid_length*grid_length*covar_Q_size[0] * covar_Q_size[1] * covar_Q_size[2] * covar_Q_size[3])
            # =======================

    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None, compute_grid_var=False):
        # Note the one_hot_grid_label is not one hot code, it is the B x N query label, the value range is 0~L*L-1

        # transform B x N x D features
        (B, N, D) = x.shape
        L = self.grid_length

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class_u = self.linear_grid_class(out)  # (B*N) * (L*L)
        # grid_class_u_np = grid_class_u.cpu().detach().numpy()
        # grid_class_u_np2 = torch.log_softmax(grid_class_u, dim=1).cpu().detach().numpy()

        if self.cls_uncertainty == True:
            # apply heteroscedastic uncertainty over classification, using Monte Carlo simulation to
            # estimate the probability output
            T = self.sample_times  # 30
            rho = self.linear_class_uncertainty(out)  # (B*N) * (L*L)

            if self.sigma_computing_mode == 0:  # sigma = log(1+exp(x))
                sigmas = torch.nn.functional.softplus(rho)# (B*N) * (L*L)
            elif self.sigma_computing_mode == 1:  # x = log(sigma^2)
                sigmas = torch.exp(rho/2)
            elif self.sigma_computing_mode == 2:  # sigma = alpha * sigmoid(x) + beta
                sigmas = self.alpha * torch.sigmoid(rho) + self.beta
            elif self.sigma_computing_mode == 3:  # max_{a}(x, 0) = (x + sqrt(x^2 + a))/2
                sigmas = (rho + torch.sqrt(rho ** 2 + self.beta)) / 2

            # draw image for grid sigmas
            # sigmas_np = sigmas.cpu().detach().numpy()
            # sigma_ims = make_uncertainty_map(sigmas_np, B)  # B x 368 x 368
            # sigma_ims_in_one = make_grid_images(torch.Tensor(sigma_ims).reshape(B, 1, 368, 368), denormalize=False)
            # sigma_ims_in_one = sigma_ims_in_one[:, :, 0].numpy()
            # save_plot_image(sigma_ims_in_one, 'sigma.png', does_show=False)

            sample_z = sampler(grid_class_u, sigmas, sample_times=T, set_negative_noise=self.negative_noise) # (B*N) * (L*L) * T
            # sample_z_np = sample_z.cpu().detach().numpy()
            grid_class_prob = sample_z.softmax(dim=1).mean(dim=2)  # Monte Carlo Simulation, (B*N) * (L*L)
            # grid_class_prob_np = (grid_class_prob).cpu().detach().numpy()
            grid_class = torch.log(grid_class_prob + 1e-9)
            # grid_class_np = grid_class.cpu().detach().numpy()

            # compute Variance var(g)
            if compute_grid_var:
                grid_var = compute_location_var(sample_z.cpu().detach(), self.grid_length, T, method=2)  # (B*N) * 2
                grid_var = grid_var.reshape(B, N, 2)

        else:
            grid_class = torch.log_softmax(grid_class_u, dim=1)
            if compute_grid_var:  # no grid var for such situation
                grid_var = None


        grid_deviation_out = self.linear_grid_deviation(out)            # (B * N) * (L*L*2)
        # grid_deviation_out = grid_deviation_out.reshape(B * N, L*L, 2)  # (B * N) * (L*L) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.

        if training_phase == True:
            # kp_label = one_hot_grid_label.reshape(B * N, -1)            # (B*N) * 2
            # retrieval_index = kp_label[:, 1] * L + kp_label[:, 0]       # (B*N)
            retrieval_index = one_hot_grid_label.view(-1)   # (B*N)
        else:  # testing
            # compute one-hot grid label for predicted grids
            retrieval_index = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
        ind_temp = torch.linspace(0, B * N - 1, B * N).long().view(-1).cuda()
        grid_deviation_out = grid_deviation_out.view(-1, 2)  # (B*N*L*L) * 2
        grid_deviation = grid_deviation_out[ind_temp * L* L + retrieval_index, :]  # (B*N) * 2

        if self.reg_uncertainty == True:
            # independent var for each point B x N x 2 or covariance for each point, B x N x (2d)
            if self.ureg_type == 0 or self.ureg_type == 1:
                rho_reg = self.linear_dev_uncertainty(out)  # (B * N) * (L*L*2) or (B * N) * (L*L*2d)
                # rho_reg = rho_reg.reshape(B * N, L*L, 2)
                rho_reg = rho_reg.view(B*N*L*L, -1)  # (B*N*L*L) * 2 or (B*N*L*L) * (covar_Q_size[0] * covar_Q_size[1])
                rho_reg = rho_reg[ind_temp * L * L + retrieval_index, :]  # (B*N) * 2 or (B*N) * (covar_Q_size[0] * covar_Q_size[1])
                rho_reg = rho_reg.reshape(B, N, -1)  # B x N x 2 Or B x N x 2d

            # =======================
            # covariance for all keypoints, B x (N x N x 2d), input is descriptor
            elif self.ureg_type == 2:
                out = x.reshape(B, -1)  # B x (N*C)
                rho_reg = self.linear_dev_uncertainty(out)  # B x (N*N*2d)
                rho_reg = rho_reg.reshape(B, N, N, -1)  # B x N x N x (2d), each block is 2 x d

            # covariance for all keypoints, B x (N x N x 2d), input is descritpor concatenated with one-hot code
            elif self.ureg_type == 3:
                one_hot_code = torch.zeros(B, N, self.grid_length**2).cuda()
                one_hot_code = one_hot_code.scatter(dim=2, index=retrieval_index.reshape(B, N, 1), value=1)  # B x N x (L*L)
                new_descriptors = torch.cat([one_hot_code, x], dim=2)  # B x N x (L*L + D)

                new_descriptors = new_descriptors.reshape(B, -1)  # B x (N*(D + L*L))
                rho_reg = self.linear_dev_uncertainty(new_descriptors)  # B x (N*N*2d)
                rho_reg = rho_reg.reshape(B, N, N, -1)  # B x N x N x (2d), each block is 2 x d
            # =======================


        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 2  (tx, ty)

        if self.reg_uncertainty == True and compute_grid_var == False:
            return grid_class, grid_deviation, rho_reg, None
        elif self.reg_uncertainty == True and compute_grid_var == True:
            return grid_class, grid_deviation, rho_reg, grid_var
        else:
            return grid_class, grid_deviation, None, None

class CovarNet(nn.Module):
    def __init__(self, grid_length=8, covar_Q_size=(2,2,2,3), ureg_type=3):
        super(CovarNet, self).__init__()
        linear_in_channel = 1024 * 2 * 2  # 1/32, 1/46, 1/64, needs to be tuned manually

        if ureg_type == 2: # covariance for all points, directly use descriptors, B x [N*N*2d]
            self.linear_dev_covar = nn.Linear(linear_in_channel*covar_Q_size[0], covar_Q_size[0] * covar_Q_size[1] * covar_Q_size[2] * covar_Q_size[3])
        elif ureg_type == 3: # covariance for all points, use descriptors concat with one-hot code, B x [N*N*2d]
            # concatenate covar_Q_size[0] descriptors into one for regressing the covariance among multiple keypoints
            self.linear_dev_covar = nn.Linear((grid_length * grid_length + linear_in_channel)*covar_Q_size[0], covar_Q_size[0] * covar_Q_size[1] * covar_Q_size[2] * covar_Q_size[3])

    def forward(self, x):
        # x: n x [(C+ L*L)*num_kp], combined descriptor for multiple kps
        # out: n x [num_kp * num_kp * 2 * d]
        x = self.linear_dev_covar(x)
        return x

class OffsetLearningNet(nn.Module):
    def __init__(self, patch_size: tuple, input_channel: int, num_of_patches_per_image: int, method=0):
        super(OffsetLearningNet, self).__init__()
        self.method = method
        if method == 0:  # shared conv + parallel linear layer
            # (B*T) x C x H x W
            self.conv = nn.Sequential(
                nn.Conv2d(input_channel, 2048, kernel_size=patch_size, stride=1, padding=0),
                # nn.BatchNorm2d(2048),
                nn.ReLU(True),
            )
            self.offset_heads = nn.ModuleList()
            for i in range(num_of_patches_per_image):
                self.offset_heads += [nn.Linear(2048, 2)]

            # initialization method 1
            self.conv[0].weight.data.normal_(mean=0, std=0.005)  # 0.005
            self.conv[0].bias.data.zero_()
            for i in range(num_of_patches_per_image):
                self.offset_heads[i].weight.data.normal_(mean=0, std=0.005)  # 0.005
                self.offset_heads[i].bias.data.zero_()

            # # initialization method 2
            # nn.init.kaiming_normal_(self.conv[0].weight)
            # # nn.init.zeros_(self.conv[0].bias)
            # for i in range(num_of_patches_per_image):
            #     nn.init.kaiming_normal_(self.offset_heads[i].weight)
            #     # nn.init.zeros_(self.offset_heads[i].bias)

        elif method == 1:  # parallel conv + linear layer
            self.offset_heads = nn.ModuleList()
            for i in range(num_of_patches_per_image):
                # self.offset_heads += [nn.Linear(2048, 2)]
                self.offset_heads += [nn.Sequential(
                    nn.Conv2d(input_channel, 1024, kernel_size=patch_size, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.Conv2d(1024, 2, kernel_size=1, stride=1, padding=0)
                )]
            # for i in range(num_of_patches_per_image):
            #     for j in [0, 2]:
            #         self.offset_heads[i][j].weight.data.normal_(mean=0, std=0.005)
            #         self.offset_heads[i][j].bias.data.zero_()
        elif method == 2:  # single conv + linear layer
            self.offset_heads = nn.Sequential(
                    nn.Conv2d(input_channel, 2048, kernel_size=patch_size, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.Conv2d(2048, 2, kernel_size=1, stride=1, padding=0)
                )
            for j in [0, 2]:
                self.offset_heads[j].weight.data.normal_(mean=0, std=0.005)  # 0.005
                self.offset_heads[j].bias.data.zero_()

    def forward(self, x):
        # B x T x C x H x W
        B, T, C, H, W = x.shape

        if self.method == 0:
            shared_features = self.conv(x.reshape(B*T, C, H, W))
            shared_features = shared_features.view(B, T, 2048)
            out_offsets_list = []
            for t in range(T):
                out_offsets = self.offset_heads[t](shared_features[:, t, :])  # B x 2
                out_offsets = out_offsets.reshape(B, 1, 2)  #  B x 1 x 2
                out_offsets_list.extend([out_offsets])
            out_offsets_final = torch.cat(out_offsets_list, dim=1)  # B x T x 2
        elif self.method == 1:
            out_offsets_list = []
            for t in range(T):
                out_features = self.offset_heads[t](x[:, t, :, :, :])
                out_features = out_features.view(B, 1, 2)
                out_offsets_list.extend([out_features])
            out_offsets_final = torch.cat(out_offsets_list, dim=1)  # B x T x 2
        elif self.method == 2:
            out_features = self.offset_heads(x.reshape(B * T, C, H, W))
            out_offsets_final = out_features.reshape(B, T, 2)


        return out_offsets_final


class GridBasedLocatorX3(nn.Module):
    def __init__(self, grid_length=8, reg_uncertainty=True, cls_uncertainty=True, sample_times=15, negative_noise=False, sigma_computing_mode=0, alpha=1, beta=0):
        super(GridBasedLocatorX3, self).__init__()

        self.grid_length = grid_length
        self.sample_times = sample_times
        self.negative_noise = negative_noise
        self.reg_uncertainty = reg_uncertainty
        self.cls_uncertainty = cls_uncertainty
        self.sigma_computing_mode = sigma_computing_mode
        if sigma_computing_mode == 2:
            self.alpha = alpha
            self.beta  = beta

        in_channel = 512  # (B*N) x 512 x 5 x 5
        cls_out_channel = 1  # L*L*1
        if self.cls_uncertainty == True:
            cls_out_channel = 2
        reg_out_channel = 2  # L*L*2
        if self.reg_uncertainty == True:
            reg_out_channel = 4

        # s = grid_length // 5
        # ks = grid_length + 1 - 5*s
        self.cls_low1 = nn.Sequential(
            nn.UpsamplingBilinear2d((grid_length, grid_length)),
            # nn.ConvTranspose2d(512, cls_out_channel, kernel_size=ks, stride=s, padding=0),
            nn.Conv2d(in_channel, cls_out_channel, kernel_size=1, stride=1, padding=0),
        )
        self.reg_low1 = nn.Sequential(
            nn.UpsamplingBilinear2d((grid_length, grid_length)),
            # nn.ConvTranspose2d(512, reg_out_channel, kernel_size=ks, stride=s, padding=0),
            nn.Conv2d(in_channel, reg_out_channel, kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None, compute_grid_var=False):
        # Note the one_hot_grid_label is not one hot code, it is the B x N query label, the value range is 0~L*L-1

        # B x N x C x H x W
        (B, N, C, H, W) = x.shape
        L = self.grid_length

        # (B*N) x C x H x W
        out = x.reshape(B*N, C, H, W)

        out_grid = self.cls_low1(out)  # (B*N) * 1 * L * L or (B*N) * 2 * L * L
        out_reg  = self.reg_low1(out)  # (B*N) * 2 * L * L or (B*N) * 4 * L * L

        grid_class_u = torch.zeros((B*N), 1, L, L, requires_grad=True).cuda()
        grid_class_u[:, 0, :, :] = (out_grid[:, 0, :, :])
        grid_class_u = grid_class_u.reshape(B*N, L*L)  # (B*N) * (L*L)
        # grid_class_u_np = grid_class_u.cpu().detach().numpy()
        # grid_class_u_np2 = torch.log_softmax(grid_class_u, dim=1).cpu().detach().numpy()

        if self.cls_uncertainty == True:
            # apply heteroscedastic uncertainty over classification, using Monte Carlo simulation to
            # estimate the probability output
            T = self.sample_times  # 30
            rho = torch.zeros((B*N), 1, L, L, requires_grad=True).cuda()
            rho[:, 0, :, :] = (out_grid[:, 1, :, :])
            rho = rho.reshape(B*N, L*L)   # (B*N) * (L*L)

            if self.sigma_computing_mode == 0:  # sigma = log(1+exp(x))
                sigmas = torch.nn.functional.softplus(rho)# (B*N) * (L*L)
            elif self.sigma_computing_mode == 1:  # x = log(sigma^2)
                sigmas = torch.exp(rho/2)
            elif self.sigma_computing_mode == 2:  # sigma = alpha * sigmoid(x) + beta
                sigmas = self.alpha * torch.sigmoid(rho) + self.beta
            elif self.sigma_computing_mode == 3:  # max_{a}(x, 0) = (x + sqrt(x^2 + a))/2
                sigmas = (rho + torch.sqrt(rho ** 2 + self.beta)) / 2

            # draw image for grid sigmas
            # sigmas_np = sigmas.cpu().detach().numpy()
            # sigma_ims = make_uncertainty_map(sigmas_np, B)  # B x 368 x 368
            # sigma_ims_in_one = make_grid_images(torch.Tensor(sigma_ims).reshape(B, 1, 368, 368), denormalize=False)
            # sigma_ims_in_one = sigma_ims_in_one[:, :, 0].numpy()
            # save_plot_image(sigma_ims_in_one, 'sigma.png', does_show=False)

            sample_z = sampler(grid_class_u, sigmas, sample_times=T, set_negative_noise=self.negative_noise) # (B*N) * (L*L) * T
            # sample_z_np = sample_z.cpu().detach().numpy()
            grid_class_prob = sample_z.softmax(dim=1).mean(dim=2)  # Monte Carlo Simulation, (B*N) * (L*L)
            # grid_class_prob_np = (grid_class_prob).cpu().detach().numpy()
            grid_class = torch.log(grid_class_prob + 1e-9)
            # grid_class_np = grid_class.cpu().detach().numpy()

            # compute Variance var(g)
            if compute_grid_var:
                grid_var = compute_location_var(sample_z.cpu().detach(), self.grid_length, T, method=2)  # (B*N) * 2
                grid_var = grid_var.reshape(B, N, 2)

        else:
            grid_class = torch.log_softmax(grid_class_u, dim=1)


        grid_deviation_out = torch.zeros((B*N), 2, L, L, requires_grad=True).cuda()
        grid_deviation_out[:, 0:2, :, :] = (out_reg[:, 0:2, :, :])
        grid_deviation_out = grid_deviation_out.permute(0, 2, 3, 1).reshape(B*N, L*L*2)  # (B * N) * (L*L*2)
        # grid_deviation_out = grid_deviation_out.reshape(B * N, L*L, 2)  # (B * N) * (L*L) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.

        if self.reg_uncertainty == True:
            rho_reg = torch.zeros((B*N), 2, L, L, requires_grad=True).cuda()
            rho_reg[:, 2:, :, :] = out_reg[:, 2:, :, :]
            rho_reg = rho_reg.permute(0, 2, 3, 1).reshape(B*N, L*L*2)  # (B * N) * (L*L*2)
            # rho_reg = rho_reg.reshape(B * N, L*L, 2)

        if training_phase == True:
            # kp_label = one_hot_grid_label.reshape(B * N, -1)            # (B*N) * 2
            # retrieval_index = kp_label[:, 1] * L + kp_label[:, 0]       # (B*N)
            retrieval_index = one_hot_grid_label.view(-1)   # (B*N)
            ind_temp = torch.linspace(0, B*N-1, B*N).long().view(-1).cuda()
            grid_deviation_out = grid_deviation_out.view(-1, 2)  # (B*N*L*L) * 2
            grid_deviation = grid_deviation_out[ind_temp * L* L + retrieval_index, :]  # (B*N) * 2
            if self.reg_uncertainty == True:
                rho_reg = rho_reg.view(-1, 2)  # (B*N*L*L) * 2
                rho_reg = rho_reg[ind_temp * L * L + retrieval_index, :]  # (B*N) * 2
                rho_reg = rho_reg.reshape(B, N, -1)  # B x N x 2
        else:  # testing
            # compute one-hot grid label for predicted grids
            predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
            ind_temp = torch.linspace(0, B * N - 1, B * N).long().view(-1).cuda()
            grid_deviation_out = grid_deviation_out.view(-1, 2)  # (B*N*L*L) * 2
            grid_deviation = grid_deviation_out[ind_temp * L* L + predict_grids, :]  # (B*N) * 2
            if self.reg_uncertainty == True:
                rho_reg = rho_reg.view(-1, 2)  # (B*N*L*L) * 2
                rho_reg = rho_reg[ind_temp * L * L + predict_grids, :]  # (B*N) * 2
                rho_reg = rho_reg.reshape(B, N, -1)  # B x N x 2

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 2  (tx, ty)

        if self.reg_uncertainty == True and compute_grid_var == False:
            return grid_class, grid_deviation, rho_reg, None
        elif self.reg_uncertainty == True and compute_grid_var == True:
            return grid_class, grid_deviation, rho_reg, grid_var
        else:
            return grid_class, grid_deviation, None, None

class GridBasedLocator_ShareNet(nn.Module):
    def __init__(self, grid_length=8):
        super(GridBasedLocator_ShareNet, self).__init__()

        self.grid_length = grid_length
        # self.convs = nn.Sequential()
        # self.convs.add_module('conv0', nn.Conv2d(input_channel, 1024, kernel_size=3, stride=2, padding=1)),
        # self.convs.add_module('relu0', nn.ReLU(True))
        # for i in range(conv_layers - 1):
        #     self.convs.add_module('conv%d'%(i + 1), nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)),
        #     self.convs.add_module('relu%d'%(i + 1), nn.ReLU(True))

        linear_in_channel = 1024 * 2 * 2  # needs to be tuned manually
        self.linear_grid_class = nn.Sequential(
            nn.Linear(linear_in_channel, grid_length * grid_length),
            nn.LogSoftmax(dim=1)
        )

        self.linear_share = nn.Sequential(
            nn.Linear(grid_length * grid_length + linear_in_channel, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
        )
        self.linear_grid_deviation = nn.Sequential(
            nn.Linear(1024, 2)
        )

        self.linear_dev_uncertainty = nn.Sequential(
            nn.Linear(1024, 2)
        )
        # self.linear_grid_deviation = nn.Sequential(
        #     nn.Linear(grid_length * grid_length + linear_in_channel, 2)
        # )
        #
        # self.linear_dev_uncertainty = nn.Sequential(
        #     nn.Linear(grid_length * grid_length + linear_in_channel, 2)
        # )


    def forward(self, x: torch.Tensor, training_phase=True, one_hot_grid_label=None):
        # transform B x N x D features
        (B, N, D) = x.shape

        # B x N x D --> (B*N) x D
        out = x.reshape(B*N, -1)

        grid_class = self.linear_grid_class(out)  # (B*N) * (L*L)

        if training_phase == True:
            one_hot_grid_label = one_hot_grid_label.reshape(B*N, -1)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
        else:  # testing
            # compute one-hot grid label for predicted grids
            predict_grids = torch.max(grid_class, dim=1)[1]  # (B2 * N), 0 ~ grid_length * grid_length - 1
            one_hot_grid_label = torch.zeros(B * N, self.grid_length**2).cuda()
            one_hot_grid_label = one_hot_grid_label.scatter(dim=1, index=torch.unsqueeze(predict_grids, dim=1), value=1)  # (B2 * N) x (grid_length * grid_length)
            # temp = one_hot_grid_label.cpu().detach().numpy()
            # print(temp)
            out = torch.cat([one_hot_grid_label, out], dim=1)  # (B*N) * (L*L + D), L is grid_length
            # temp = out.cpu().detach().numpy()
            # print(temp)
        out = self.linear_share(out)
        grid_deviation = self.linear_grid_deviation(out)  # (B * N) * 2
        # IMPORTANT!!!
        # grid_deviation = torch.sigmoid(grid_deviation)  # used when deviation range is 0~1, otherwise it should be commented.
        rho = self.linear_dev_uncertainty(out)

        grid_class = grid_class.reshape(B, N, -1)  # B x N x (L * L), where L is the desired linear_out_channel
        grid_deviation = grid_deviation.reshape(B, N, -1)  # B x N x 4  (tx, ty, rho_x, rho_y)
        rho = rho.reshape(B, N, -1)

        return grid_class, grid_deviation, rho

class PatchUncertaintyModule(nn.Module):
    def __init__(self, input_channel=1024, method=0, conv_layers=1):
        super(PatchUncertaintyModule, self).__init__()
        self.net = nn.Sequential()
        if conv_layers == 1:
            self.net.add_module('conv1d_0', nn.Conv1d(input_channel, 1, kernel_size=1, padding=0))
        elif conv_layers == 2:
            self.net.add_module('conv1d_0', nn.Conv1d(input_channel, 1024, kernel_size=1, padding=0))
            # self.net.add_module('bn_0', nn.BatchNorm1d(1024)),
            self.net.add_module('relu_0', nn.ReLU(True))
            self.net.add_module('conv1d_1', nn.Conv1d(1024, 1, kernel_size=1, padding=0))
        elif conv_layers == 3:
            self.net.add_module('conv1d_0', nn.Conv1d(input_channel, 1024, kernel_size=1, padding=0))
            # self.net.add_module('bn_0', nn.BatchNorm1d(1024)),
            self.net.add_module('relu_0', nn.ReLU(True))
            self.net.add_module('conv1d_1', nn.Conv1d(1024, 512, kernel_size=1, padding=0))
            # self.net.add_module('bn_1', nn.BatchNorm1d(512)),
            self.net.add_module('relu_1', nn.ReLU(True))
            self.net.add_module('conv1d_2', nn.Conv1d(512, 1, kernel_size=1, padding=0))


    def forward(self, x: torch.Tensor):
        '''
        feed x to fully convolutional network
        :param x: B x C x N
        :return: out, B x N, each fiber is transformed into a scalar value w
        '''
        B, _, N = x.shape
        out = self.net(x)      # B x 1 x N
        out = out.reshape(B, N)  # B x N
        return out

## =======================================================
## below codes are not used
## =======================================================
class ResNet34BasedModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34BasedModel, self).__init__()

        self.resnet = resnet34(pretrained=True)
        # 卷积层
        self.backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            # nn.AvgPool2d(7, stride=1)
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # 增加的全连接层
        self.fc_adapt = nn.Sequential(
            # 将512 x 8 x 8的特征图转化为512维特征向量
            nn.Linear(512 * 8 * 8, 512),
            # nn.ReLU(True) # ReLU会导致特征消失，而Sigmoid不会
            nn.Sigmoid()
        )
        self.fc_adapt[0].weight.data.normal_(0, 0.5)

        # 最后的分类层
        self.class_classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # domain classifier which is used for adversarial domain training
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(512, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data: torch.Tensor):
        """前向传播
        
        Arguments:
            input_data {torch.Tensor} -- 样本输入 batch × 3 × height × width
            input_data image should be resized to 300 x 300 becasue the Linear connected layers are designed on this basis.
        Returns:
            [torch.Tensor] -- 最后一个全连接层的样本分类输出
            [torch.Tensor] -- 域分类输出
            [torch.Tensor] -- 深度特征输出
        """
        # if input_target is not None:
        #     input_source, input_target = self.features(input_source), self.features(input_target)
        #     input_source, input_target = input_source.view(input_source.size(0), -1), input_target.view(input_target.size(0), -1)
        #     input_source, input_target = self.fc(input_source), self.fc(input_target)
        #     return input_source, input_target, self.classifier(input_source), self.classifier(input_target)
        # else:
        #     input_source = self.features(input_source)
        #     return self.classifier(self.fc(input_source.view(input_source.size(0), -1)))
        feature = self.backbone(input_data)
        feature = self.fc_adapt(feature.view(feature.size(0), -1))
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)
        return class_output, domain_output, feature


class ResNet34BasedModel_GAP(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34BasedModel, self).__init__()

        self.resnet = resnet34(pretrained=True)
        # 卷积层
        self.resnet_backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            # nn.AvgPool2d(7, stride=1)
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d((8, 8)),
            nn.Sigmoid()
        )

        # # 增加的全连接层
        # self.fc = nn.Sequential(
        #     # 将512 x 8 x 8的特征图转化为512维特征向量
        #     nn.Linear(512 * 8 * 8, 512),
        #     # nn.ReLU(True) # ReLU会导致特征消失，而Sigmoid不会
        #     nn.Sigmoid()
        # )
        # self.fc[0].weight.data.normal_(0, 0.5)

        # 最后的分类层
        self.class_classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # domain classifier which is used for adversarial domain training
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(512, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data: torch.Tensor):
        """前向传播

        Arguments:
            input_data {torch.Tensor} -- 样本输入 batch × 3 × height × width

        Returns:
            [torch.Tensor] -- 最后一个全连接层的样本分类输出
            [torch.Tensor] -- 域分类输出
            [torch.Tensor] -- 深度特征输出
        """
        # if input_target is not None:
        #     input_source, input_target = self.features(input_source), self.features(input_target)
        #     input_source, input_target = input_source.view(input_source.size(0), -1), input_target.view(input_target.size(0), -1)
        #     input_source, input_target = self.fc(input_source), self.fc(input_target)
        #     return input_source, input_target, self.classifier(input_source), self.classifier(input_target)
        # else:
        #     input_source = self.features(input_source)
        #     return self.classifier(self.fc(input_source.view(input_source.size(0), -1)))
        feature = self.resnet_backbone(input_data)
        # feature = self.fc(feature.view(feature.size(0), -1))
        feature = self.gap(feature)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)
        return class_output, domain_output, feature


class ResNet50BasedModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50BasedModel, self).__init__()

        self.resnet = resnet50(pretrained=True)
        # # 卷积层
        self.backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            nn.AvgPool2d(7, stride=1, padding=0)
            # nn.AdaptiveAvgPool2d((8, 8))
        )

        # 增加的全连接层
        self.fc_adapt = nn.Sequential(
            # 将2048 x 4 x 4的特征图转化为512维特征向量
            nn.Linear(2048 * 4 * 4, 512),
            # nn.ReLU(True) # ReLU会导致特征消失，而Sigmoid不会
            nn.Sigmoid()
        )
        self.fc_adapt[0].weight.data.normal_(0, 0.5)

        # 最后的分类层
        self.class_classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # domain classifier which is used for adversarial domain training
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(512, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data: torch.Tensor):
        """前向传播

        Arguments:
            input_data {torch.Tensor} -- 样本输入 batch × 3 × height × width

        Returns:
            [torch.Tensor] -- 最后一个全连接层的样本分类输出
            [torch.Tensor] -- 域分类输出
            [torch.Tensor] -- 深度特征输出
        """
        # if input_target is not None:
        #     input_source, input_target = self.features(input_source), self.features(input_target)
        #     input_source, input_target = input_source.view(input_source.size(0), -1), input_target.view(input_target.size(0), -1)
        #     input_source, input_target = self.fc(input_source), self.fc(input_target)
        #     return input_source, input_target, self.classifier(input_source), self.classifier(input_target)
        # else:
        #     input_source = self.features(input_source)
        #     return self.classifier(self.fc(input_source.view(input_source.size(0), -1)))
        feature = self.backbone(input_data)
        feature = self.fc_adapt(feature.view(feature.size(0), -1))
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)
        return class_output, domain_output, feature


class AlexNetBasedModel(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetBasedModel, self).__init__()
        self.backbone = nn.Sequential(
            alexnet(pretrained=True).features,
            # fc6
            nn.Linear(256 * 8 * 8, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # fc7
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout()
        )
        # 增加的全连接层
        self.fc_adapt = nn.Sequential(
            # 将256 x 8 x 8的特征图转化为512维特征向量
            nn.Linear(4096, 512),
            nn.ReLU(True) # ReLU会导致特征消失，而Sigmoid不会
            # nn.Sigmoid()
        )
        self.fc_adapt[0].weight.data.normal_(0, 0.1)
        self.class_classifier = nn.Sequential(
            # fc8
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            # nn.Linear(256, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, input_data : torch.Tensor):
        # print(self.alexnet_backbone)
        for name, module in self.backbone._modules.items():
            if name == '0':
                feature = module(input_data)
                feature = feature.view(feature.size(0), -1)
            else:
                feature = module(feature)
        feature = self.fc_adapt(feature)  # adaptation layer
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)
        return class_output, domain_output, feature





class AlexNetBasedModel_GRL(nn.Module):
    def __init__(self, num_classes):

        super(AlexNetBasedModel, self).__init__()
        self.feature = alexnet(pretrained=True).features

        self.fc6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 8 * 8, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )
        self.fc7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
        )
        self.fc_adapt = nn.Sequential(
            nn.Linear(4096, 256),
            # nn.Sigmoid()
            nn.ReLU(inplace=True)
        )
        self.fc_adapt[0].weight.data.normal_(0, 0.1)
        self.fc8 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
            nn.ReLU(inplace=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data: torch.autograd.Variable, alpha: float = 0):
        """前向传播
        Arguments:
        """
        feature = self.feature(input_data)
        feature = feature.view(-1, 256 * 8 * 8)
        feature = self.fc7(self.fc6(feature))
        feature = self.fc_adapt(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.fc8(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output, feature

class VGG16BasedModel(nn.Module):
    def __init__(self, num_classes):
        super(VGG16BasedModel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 9 * 9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)  # default 0.5
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc_adapt = nn.Sequential(
            nn.Linear(4096, 512),
            # nn.ReLU()  # exploded
            # nn.Softplus() # exploded
            nn.Sigmoid()
        )
        nn.init.normal_(self.fc1[0].weight, 0, 0.1)
        nn.init.normal_(self.fc2[0].weight, 0, 0.1)
        nn.init.normal_(self.fc_adapt[0].weight, 0, 0.1)

        self.backbone = nn.Sequential(
            vgg16(pretrained=True).features,
            self.fc1,
            self.fc2
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            # nn.Linear(256, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, input_data: torch.Tensor):
        # for name, module in self.backbone._modules.items():
        #     if name == '0':
        #         feature = module(input_data)
        #         feature = feature.view(feature.size(0), -1)
        #     else:
        #         feature = module(feature)
        feature = self.backbone[0](input_data)
        feature = self.backbone[1](feature.view(feature.size(0), -1))
        feature = self.backbone[2](feature)
        feature = self.fc_adapt(feature)  # adaptation layer
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)
        return class_output, domain_output, feature


# this CNNModel is used for handwritten digit recognition, 10 classes
class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        self.conv = nn.Conv2d(3, 32, kernel_size=(3, 3))

    def forward(self, input_data: torch.Tensor, alpha: float = 0):
        if input_data.data.shape[1] < 3:
            input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28) #expand单通道图像至三通道
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        # reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        # domain_output = self.domain_classifier(reverse_feature)
        domain_output = self.domain_classifier(feature)

        return class_output, domain_output, feature

# This model used for SVHN
class CNNModel2(nn.Module):

    def __init__(self):
        super(CNNModel2, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(64))
        self.feature.add_module('f_drop2', nn.Dropout2d())
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('f_conv3', nn.Conv2d(64, 128, kernel_size=5, stride=1))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(128))
        self.feature.add_module('f_drop3', nn.Dropout2d())
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_pool3', nn.MaxPool2d(kernel_size=2))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(128 * 2 * 2, 3072))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(3072))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(3072, 2048))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(2048))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(2048, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(128 * 2 * 2, 1024))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(1024))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(1024))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(1024, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data: torch.Tensor, alpha: float = 0):
        if input_data.data.shape[1] < 3:
            input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32) #expand单通道图像至三通道
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 2 * 2)
        # reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        # domain_output = self.domain_classifier(reverse_feature)
        domain_output = self.domain_classifier(feature)

        return class_output, domain_output, feature


# this CNNModel is used for handwritten digit recognition, 10 classes
class CNNModelA(nn.Module):

    def __init__(self):
        super(CNNModelA, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 32, kernel_size=5, stride=1))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(32))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1)))
        self.feature.add_module('f_conv2', nn.Conv2d(32, 48, kernel_size=5, stride=1))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(48))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1)))
        self.feature.add_module('f_drop2', nn.Dropout())

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(48 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_drop2', nn.Dropout2d())
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(48 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


    def forward(self, input_data: torch.Tensor, alpha: float = 0):
        if input_data.data.shape[1] < 3:
            input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28) #expand单通道图像至三通道
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        # reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        # domain_output = self.domain_classifier(reverse_feature)
        domain_output = self.domain_classifier(feature)

        return class_output, domain_output, feature

# This model used for SVHN
class CNNModelB(nn.Module):

    def __init__(self):
        super(CNNModelB, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(64))
        # self.feature.add_module('f_drop2', nn.Dropout2d())
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('f_conv3', nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(128))
        # self.feature.add_module('f_drop3', nn.Dropout2d())
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_fc4', nn.Linear(8192, 3072))
        self.feature.add_module('f_bn4', nn.BatchNorm1d(3072))
        self.feature.add_module('f_relu4', nn.ReLU(True))
        self.feature.add_module('f_drop4', nn.Dropout())

        self.class_classifier = nn.Sequential()
        # self.feature.add_module('c_fc1', nn.Linear(8912, 3072))
        # self.feature.add_module('c_bn1', nn.BatchNorm1d(3072))
        # self.feature.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(3072, 2048))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(2048))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(2048, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(3072, 1024))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(1024))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(1024))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(1024, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data: torch.Tensor, alpha: float = 0):
        if input_data.data.shape[1] < 3:
            input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32) #expand单通道图像至三通道
        for name, module in self.feature._modules.items():
            if name == 'f_conv1':
                feature = module(input_data)
            elif name == 'f_relu3':
                feature = module(feature)
                feature = feature.view(feature.size(0), -1)
                # print(feature.shape)
            else:
                feature = module(feature)
        # feature = self.feature(input_data)
        # feature = feature.view(-1, 128 * 2 * 2)
        # reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        # domain_output = self.domain_classifier(reverse_feature)
        domain_output = self.domain_classifier(feature)

        return class_output, domain_output, feature


