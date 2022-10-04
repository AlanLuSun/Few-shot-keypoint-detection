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
from network.models_gridms2 import average_representations2

import math
import time

from heatmap import putGaussianMaps
import copy
import numpy as np


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, feature_channel, hidden_size, mid_layer_num=0):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(feature_channel, hidden_size, kernel_size=1, padding=0),
                        nn.BatchNorm2d(hidden_size, momentum=1, affine=True),
                        nn.ReLU(),
                        # nn.MaxPool2d(2)
        )
        self.mid_layer_num = mid_layer_num
        if self.mid_layer_num > 0:  # middle repetitive layers
            self.mid_layers = nn.Sequential()
            for i in range(self.mid_layer_num):
                self.mid_layers.add_module('m_conv{}'.format(i), nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0))
                self.mid_layers.add_module('m_bn{}'.format(i), nn.BatchNorm2d(hidden_size, momentum=1, affine=True))
                self.mid_layers.add_module('m_relu{}'.format(i), nn.ReLU())

        self.layer2 = nn.Conv2d(hidden_size, 1, kernel_size=1, padding=0)

    def forward(self,x):
        # x: (B2 * N) x 2C x H x W
        out = self.layer1(x)
        if self.mid_layer_num > 0:  # middle repetitive layers
            out = self.mid_layers(out)
        out = self.layer2(out)
        out = torch.sigmoid(out)
        return out


class LinearDiag(nn.Module):
    def __init__(self, feature_channel, init_value=1./1000, bias=False):
        super(LinearDiag, self).__init__()
        # weight = torch.FloatTensor(num_features).fill_(1.) # initialize to the identity transform
        weight = torch.FloatTensor(feature_channel).fill_(init_value) # tensor size: C, initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(feature_channel).fill_(0)  # tensor size: C (feature_channel)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        '''
        X: N (N features) x C (feature_channel)
        '''
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)  # channel-wise multiplication (x \odot w)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out

class LinearMat(nn.Module):
    def __init__(self, feature_channel, init_value=1./1000, bias=True, num_layers=1):
        super(LinearMat, self).__init__()
        if num_layers == 1:
            self.fc_blk = nn.Sequential()
            self.fc_blk.add_module('fc', nn.Linear(feature_channel, feature_channel, bias=bias))
            # self.fc_blk[0].weight.data.copy_(torch.eye(feature_channel, feature_channel) + torch.randn(feature_channel, feature_channel)*0.001)
            self.fc_blk[0].weight.data.copy_(torch.eye(feature_channel, feature_channel) * init_value + torch.randn(feature_channel, feature_channel) * init_value * 0.001)
            # self.fc.bias.data.fill_(0)
            self.fc_blk[0].bias.data.zero_()
        elif num_layers > 1:
            self.fc_blk = nn.Sequential()
            for i in range(num_layers-1):
                self.fc_blk.add_module('fc{}'.format(i), nn.Linear(feature_channel, feature_channel, bias=bias))
                self.fc_blk.add_module('relu{}'.format(i), nn.ReLU())
            self.fc_blk.add_module('fc{}'.format(num_layers-1), nn.Linear(feature_channel, feature_channel, bias=bias))
            # initialization
            for i in range(num_layers-1):
                self.fc_blk[i].weight.data.normal_(mean=0, std=0.01)
                self.fc_blk[i].bias.data.zero_()
            self.fc_blk[(num_layers-1)*2].weight.data.copy_(torch.eye(feature_channel, feature_channel) * init_value + torch.randn(feature_channel, feature_channel) * init_value * 0.001)
            self.fc_blk[(num_layers-1)*2].bias.data.zero_()

    def forward(self, X: torch.Tensor) -> "predicted parameters":
        '''
        X: N (N features) x C (feature_channel)
        '''
        C = self.fc_blk[0].weight.data.shape[0]
        assert X.dim() == 2 and X.size(1) == C

        out = self.fc_blk(X)  # matrix multiplication (x * w)
        # out /= C  # normalize
        return out


class AttentionBasedBlock(nn.Module):
    def __init__(self, feature_channel, num_base, scale_att=None):
        super(AttentionBasedBlock, self).__init__()
        self.feature_channel = feature_channel
        self.num_base = num_base
        self.query_layer = nn.Linear(feature_channel, feature_channel)
        self.query_layer.weight.data.copy_(torch.eye(feature_channel, feature_channel) + torch.randn(feature_channel, feature_channel) * 0.001)
        self.query_layer.bias.data.zero_()

        keys = torch.FloatTensor(num_base, feature_channel).normal_(0., np.sqrt(2.0 / feature_channel))
        self.learnable_keys = nn.Parameter(keys, requires_grad=True)

        self.use_scale_att = False
        if scale_att is not None:
            self.use_scale_att = True
            self.scale_att = nn.Parameter(torch.FloatTensor(1).fill_(scale_att), requires_grad=True)  # one learnable scalar

    def forward(self, features, base_weights):
        '''
        features: N (N features) x C (feature_channel) or B (samples) x N (N features) x C (feature_channel)
        base_weights: N_base x C (feature_channel)
        '''
        dim = features.dim()

        if dim == 2:
            N, C = features.shape
            N_base, _ = base_weights.shape
            assert N_base == self.num_base, "The number of base categories triggers error in Class AttentionBasedBlock"

            q = self.query_layer(features)  # N x C
            q_normalize = torch.nn.functional.normalize(q, p=2, dim=1, eps=1e-12)  # N x C
            keys_normalize = torch.nn.functional.normalize(self.learnable_keys, p=2, dim=1, eps=1e-12)  # N_base x C
            keys_normalize = keys_normalize.transpose(1, 0)
            if self.use_scale_att == False:  # apply learnable temperature
                attention_coefficients = torch.mm(q_normalize, keys_normalize)  # N x N_base
            else:
                attention_coefficients = self.scale_att * torch.mm(q_normalize, keys_normalize)  # N x N_base
            attention_coefficients = torch.softmax(attention_coefficients, dim=1)
            weights_novel = torch.mm(attention_coefficients, base_weights)  # N x C (feature_channel)

        elif dim == 3:
            B, N, C = features.shape
            N_base, _ = base_weights.shape
            assert N_base == self.num_base, "The number of base categories triggers error in Class AttentionBasedBlock"

            features = features.reshape(B * N, C)
            q = self.query_layer(features)
            q = q.reshape(B, N, C)
            q_normalize = torch.nn.functional.normalize(q, p=2, dim=2, eps=1e-12)  # B x N x C
            keys_normalize = torch.nn.functional.normalize(self.learnable_keys, p=2, dim=1, eps=1e-12)  # N_base x C
            keys_normalize = keys_normalize.transpose(1, 0).unsqueeze(dim=0).repeat(B, 1, 1)  # B x C x N_base
            if self.use_scale_att == False:  # apply learnable temperature
                attention_coefficients = torch.bmm(q_normalize, keys_normalize)  # B x N x N_base
            else:
                attention_coefficients = self.scale_att * torch.bmm(q_normalize, keys_normalize)  # B x N x N_base
            attention_coefficients = torch.softmax(attention_coefficients, dim=attention_coefficients.dim()-1)  # B x N x N_base
            weights_novel = torch.matmul(attention_coefficients, base_weights)  # B x N x C
        else:
            raise ValueError('Unexpected dim in input tensor. Error in AttentionBasedBlock.')

        return weights_novel


class WeightGeneratorWithAtt(nn.Module):
    def __init__(self, feature_channel, num_base, linear_type='diag', init_value=1.0/1000, scale_att=None):
        '''
        feature_channel, linear_type, init_value are for feature averaging based weight generator,
        feature_channel, num_base, scale_att are for attention based weight generator.
        '''
        super(WeightGeneratorWithAtt, self).__init__()
        # w' = \phi_1 * z_{avg} + \phi_2 * w_{att}
        # w_{att} = E_{i}E_{b} Att(\phi_q * z_i, k_b)*w_b
        if linear_type == 'diag':
            self.wg_feature_avg = LinearDiag(feature_channel, init_value)
            self.linear_att = LinearDiag(feature_channel, init_value)
        elif linear_type == 'mat':
            self.wg_feature_avg = LinearMat(feature_channel, init_value)
            self.linear_att = LinearDiag(feature_channel, init_value)
        else:
            raise ValueError('Wrong linear type for WG. Error in WeightGeneratorWithAtt')

        self.wg_att = AttentionBasedBlock(feature_channel, num_base, scale_att)

    def forward(self, avg_features, base_weights, support_kp_features=None, support_kp_mask=None):
        '''
        avg_features: N (N features) x C (feature_channel)
        base_weights: N_base x C (feature_channel)
        support_kp_features:
            if none, it means using the avg_features as statistic to get attention-based novel weights;
            else, it should has size of
            B (samples) x N (N features) x C (feature_channel),
            and we use support_kp_features to get attention-based novel weights.
        support_kp_mask: B (samples) x N (N features)
        '''
        # first part of weights is predicted from averaged features (support keypoint prototype)
        weights_novel1 = self.wg_feature_avg(avg_features)

        use_avg_feature_for_att = True
        if (support_kp_features is not None) and (support_kp_mask is not None):
            use_avg_feature_for_att = False

        # second part of weights is predicted based on attention
        if use_avg_feature_for_att == True:
            weights_novel2 = self.wg_att(avg_features, base_weights)  # N x C (feature_channel)
        else:
            weights_novel2 = self.wg_att(support_kp_features, base_weights)  # B x N x C (feature_channel)
            weights_novel2 = average_representations2(weights_novel2.transpose(2, 1), support_kp_mask)  # C x N
            weights_novel2 = weights_novel2.transpose(1, 0)  # N x C

        weights_novel = weights_novel1 + self.linear_att(weights_novel2)

        return weights_novel



# dynamic localization network whose weights are predicted from weight generator
class LocalizationNet(nn.Module):
    def __init__(self):
        super(LocalizationNet, self).__init__()
        # self.conv = nn.Conv2d(2048, 11, 1, 1, 0)
        # self.conv.weight.data.normal_(0, 0.01)

    def forward(self, feature: torch.Tensor, weights:torch.Tensor):
        '''
        feature: B x C x H x W
        dynamic weights: N x C (N_landmarks x weight_vector_dim)
        '''
        B, C, H, W = feature.shape
        N, _ = weights.shape
        # weights_np = weights.cpu().detach().numpy()
        # w2_np = self.conv.weight.data.squeeze().cpu().detach().numpy()
        weights_ext = weights.unsqueeze(dim=0).repeat(B, 1, 1)
        out = torch.bmm(weights_ext, feature.view(B, C, -1))
        out = out.reshape(B, N, H, W)

        # out2 = self.conv(feature)
        # print('ok')

        return out

# base localization network whose weights are fixed
class LocalizationNetBase(nn.Module):
    def __init__(self, num_base, feature_channel=2048, normalize_weights=False):
        super(LocalizationNetBase, self).__init__()
        self.num_base = num_base
        self.conv = nn.Conv2d(feature_channel, num_base, kernel_size=1, stride=1, padding=0, bias=False)  # no biase I set
        self.conv.weight.data.normal_(0, np.sqrt(2.0 / feature_channel))

        # weather normalize weights during feedforward inference
        self.normalize_weights = normalize_weights

    def forward(self, feature: torch.Tensor):
        '''
        feature: B x C x H x W
        '''
        if self.normalize_weights == False:
            out = self.conv(feature)
        else:
            N = self.num_base
            B, C, H, W = feature.shape
            weights = self.conv.weight #.data
            assert weights.shape[0] == N and weights.shape[1] == C, "Feature dim is not right. Error in LocalizationNetBase."
            weights = weights.reshape(N, C)
            weights = torch.nn.functional.normalize(weights, p=2, dim=1, eps=1e-12)  # N x C
            out = torch.bmm(weights.unsqueeze(0).repeat(B, 1, 1), feature.view(B, C, H*W))
            out = out.view(B, N, H, W)

        return out

# =========================================================================
# below is reference code
class RelationNetwork0(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork0, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.sigmoid(self.fc2(out))
        return out