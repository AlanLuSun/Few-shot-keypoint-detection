# coding=utf-8

import random
import sys

import cv2

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage

"""Implement the generate of every channel of ground truth heatmap.
:param centerA: int with shape (2,), every coordinate of person's keypoint.
:param accumulate_confid_map: one channel of heatmap, which is accumulated, 
       1/100 is the smallest value of heatmap (namely log(100)=d2/(2sigma^2)=4.6052).
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""


def putGaussianMaps(center, sigma, grid_y, grid_x, stride, normalization=False, accumulate_confid_map=None, use_cuda=True):
    # center (x, y)
    # sigma
    start = stride / 2.0 - 0.5
    y_range = torch.Tensor([i for i in range(int(grid_y))])
    x_range = torch.Tensor([i for i in range(int(grid_x))])
    yy, xx = torch.meshgrid(y_range, x_range)
    # xx, yy = torch.meshgrid(x_range, y_range)  # xy exchange
    if use_cuda and torch.cuda.is_available():
        xx, yy = xx.cuda(), yy.cuda()
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    confid_map = torch.exp(-exponent)
    confid_map = torch.mul(mask, confid_map)

    return_flag = True
    if normalization == True:
        # print(torch.sum(confid_map).cpu().detach())
        if torch.sum(confid_map).cpu().detach().numpy() == 0:
            print(center.cpu().detach().numpy())
            return_flag = False
            # exit(0)
        # print(torch.sum(confid_map).cpu().detach().numpy())
        confid_map = confid_map / torch.sum(confid_map).item()
    if accumulate_confid_map != None:
        accumulate_confid_map += confid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

        return accumulate_confid_map

    # map = confid_map.cpu().detach().numpy()
    # plt.imshow(map)
    # plt.show()

    return confid_map, return_flag
