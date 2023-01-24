# -*- coding: utf-8 -*-
"""dataset.py
"""
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from collections import OrderedDict
import random

from datasets.dataset_utils import draw_instance, draw_skeletons, filter_keypoints, draw_markers
from datasets.transforms import Preprocess
import datasets.transforms as mytransforms

import copy
import cv2
import math
import time
import h5py
from skimage.util import regular_grid

# 15 keypoints for CUB dataset
def get_keypoints():
    keypoints = [
        'back',
        'beak',
        'belly',
        'breast',
        'crown',
        'forehead',
        'left_eye',
        'left_leg',
        'left_wing',
        'nape',
        'right_eye',
        'right_leg',
        'right_wing',
        'tail',
        'throat',
    ]
    return keypoints

# left-right symmetry
HFLIP = {
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_leg': 'right_leg',
    'right_leg': 'left_leg',
    'left_wing': 'right_wing',
    'right_wing': 'left_wing',
}

# front-back symmetry
FBFLIP = {
}

# diagonal symmetry
DFLIP = {
}

# define the paths that are going to interpolate auxiliary keypoints
def predefined_auxiliary_paths():
    keypoint_types = get_keypoints()
    # path1 = [
    #     [keypoint_types.index('nose'), keypoint_types.index('l_ear')],
    #     [keypoint_types.index('nose'), keypoint_types.index('r_ear')],
    #     [keypoint_types.index('l_f_leg'), keypoint_types.index('l_f_paw')],
    #     [keypoint_types.index('r_f_leg'), keypoint_types.index('r_f_paw')],
    #     [keypoint_types.index('l_b_leg'), keypoint_types.index('l_b_paw')],
    #     [keypoint_types.index('r_b_leg'), keypoint_types.index('r_b_paw')],
    # ]

    path1 = [
        [keypoint_types.index('beak'), keypoint_types.index('crown')],
        [keypoint_types.index('beak'), keypoint_types.index('nape')],
        # [keypoint_types.index('crown'), keypoint_types.index('nape')],
        # [keypoint_types.index('nape'), keypoint_types.index('back')],
        [keypoint_types.index('nape'), keypoint_types.index('tail')],
        [keypoint_types.index('back'), keypoint_types.index('tail')],
        # [keypoint_types.index('back'), keypoint_types.index('throat')],
        [keypoint_types.index('back'), keypoint_types.index('breast')],
        [keypoint_types.index('back'), keypoint_types.index('belly')],
        # [keypoint_types.index('back'), keypoint_types.index('left_leg')],
        # [keypoint_types.index('back'), keypoint_types.index('right_leg')],
        # [keypoint_types.index('tail'), keypoint_types.index('breast')],
        # [keypoint_types.index('tail'), keypoint_types.index('throat')],
        # [keypoint_types.index('throat'), keypoint_types.index('breast')],
        # [keypoint_types.index('breast'), keypoint_types.index('belly')],
        # [keypoint_types.index('belly'), keypoint_types.index('left_wing')],
        # [keypoint_types.index('belly'), keypoint_types.index('right_wing')],
        # [keypoint_types.index('left_wing'), keypoint_types.index('left_leg')],
        # [keypoint_types.index('right_wing'), keypoint_types.index('right_leg')]
    ]

    return path1

def get_auxiliary_paths(path_mode='predefined', support_keypoint_categories=None, num_random_paths=6):
    # when using path_mode 'exhaust', it will perform to use all exhaustive paths, where the end points are designated from
    # specified_keypoint_categories
    if path_mode == 'exhaust' or path_mode == 'random':
        body_part_paths = []
        N = len(support_keypoint_categories)
        keypoint_types = get_keypoints()
        for i in range(0, N-1, 1):
            for j in range(i+1, N, 1):
                kp_type1 = support_keypoint_categories[i]
                kp_type2 = support_keypoint_categories[j]
                body_part_paths.append([keypoint_types.index(kp_type1), keypoint_types.index(kp_type2)])
        # print(body_part_paths)
        if path_mode == 'random':  # random choice
            total_paths = len(body_part_paths)
            if num_random_paths >= total_paths:
                pass
            else:
                body_part_paths_sampled = random.sample(body_part_paths, num_random_paths)
                body_part_paths = body_part_paths_sampled  # overwrite
    else:  # predefined
        body_part_paths_original = predefined_auxiliary_paths()
        # check whether the body_part_paths are qualified or not
        keypoint_types = get_keypoints()
        specified_kp_mask = np.zeros(len(keypoint_types))
        for kp_type in support_keypoint_categories:
            ind = keypoint_types.index(kp_type)
            specified_kp_mask[ind] = 1
        body_part_paths = []  # body_part_path is posssible to be None if none of support kps falling in the predefined paths
        for each_path in body_part_paths_original:
            if specified_kp_mask[each_path[0]] == 1 and specified_kp_mask[each_path[1]] == 1:
                body_part_paths.append(each_path)
        # print(body_part_paths)

    return body_part_paths


# global variables
KEYPOINT_TYPES = get_keypoints()
KEYPOINT_TYPE_IDS = OrderedDict(zip(KEYPOINT_TYPES, range(len(KEYPOINT_TYPES))))

def horizontal_swap_keypoints(keypoints):
    '''
    :param keypoints: a matrix of N keypoints, N x 2
    :return:
    '''
    # target = np.zeros(keypoints.shape)
    # for source_i, xyv in enumerate(keypoints):
    #     source_name = KEYPOINT_TYPES[source_i]
    #     target_name = HFLIP.get(source_name)
    #     if target_name:  # not None
    #         target_i = KEYPOINT_TYPES.index(target_name)
    #     else:  # None
    #         target_i = source_i
    #     target[target_i] = xyv

    target = copy.deepcopy(keypoints)
    # swap the position of keypoints in the first len(KEYPOINT_TYPS) of input keypoints, the position of remaining interpolated
    # keypoints are not changed.
    for source_i, source_name in enumerate(KEYPOINT_TYPES):
        xyv = keypoints[source_i]
        target_name = HFLIP.get(source_name)
        if target_name:  # not None
            target_i = KEYPOINT_TYPES.index(target_name)
        else:  # None
            target_i = source_i
        target[target_i] = xyv

    return target

def get_symmetry_keypoints(symmetry_map, keypoint_types, keypoints, query_kp_mask, union_support_kp_mask):
    '''
    :param symmetry_map: HFLIP, FBFLIP, DFLIP
    :param keypoint_types: list, N
    :param keypoints: B x N x 2
    :param support_kp_mask: B x N
    :param query_kp_mask: B x N
    :return: symmetry_keypoints, symmetry_kp_mask
    '''
    B = keypoints.shape[0]
    symmetry_keypoints = copy.deepcopy(keypoints)
    symmetry_kp_mask = copy.deepcopy(query_kp_mask)
    for source_i, source_name in enumerate(keypoint_types):
        target_name = symmetry_map.get(source_name)
        # get symmetric keypoints
        if target_name:  # not None
            target_i = keypoint_types.index(target_name)
            symmetry_keypoints[:, source_i] = keypoints[:, target_i]
        # set symmetric keypoint's mask
        for image_i in range(B):
            # exist symmetric keypoints, exist support category, exist query keypoints
            if target_name != None and query_kp_mask[image_i, target_i] and union_support_kp_mask[source_i] == 1:
                symmetry_kp_mask[image_i, source_i] = 1
            else:
                symmetry_kp_mask[image_i, source_i] = 0

    return symmetry_keypoints, symmetry_kp_mask

def construct_interpolated_keypoints(paths, full_label, interpolated_knots):
    '''
    return a numpy matrix which consists of interpolated keypoints for each given path
    :param paths: list, N_path x 2, [[path0_kp1_index, path0_kp2_index], [path1_kp1_index, path1_kp2_index], ...]
    :param full_label: all of the annotated kps, numpy, N_fullkps x 3, each row is (x, y, is_visible)
    :param interpolated_knots: numpy, a series of values in [0, 1], for instance, [0.25, 0.5, 1]
    :return:
             interpolated_kps, numpy, (N_path * N_knots) x 2;
             interpolated_kps_mask, numpy, (N_path * N_knots)
    '''
    N_path = len(paths)
    N_knots = len(interpolated_knots)
    N_kps = N_path*N_knots
    interpolated_kps = np.zeros((N_kps, 2))
    interpolated_kp_mask = np.zeros(N_kps, dtype=np.float32)

    for i, path in enumerate(paths):
        (start_kp_ind, end_kp_ind) = path
        # judge if the start point and end point both exist
        if full_label[start_kp_ind, 2] and full_label[end_kp_ind, 2]:
            start_pt = full_label[start_kp_ind, 0:2]  # (x1, y1)
            end_pt   = full_label[end_kp_ind, 0:2]  # (x2, y2)
            # interpolation type: linear interpolation, B-spline interpolation
            # linear interpolation
            w1 = 1 - interpolated_knots
            w2 = interpolated_knots
            pts = start_pt * w1.reshape(N_knots, 1) + end_pt * w2.reshape(N_knots, 1)  # using broadcast, N_knots x 2
            interpolated_kps[(i*N_knots) : ((i+1)*N_knots), :] = pts
            interpolated_kp_mask[(i*N_knots) : ((i+1)*N_knots)] = 1

    return interpolated_kps, interpolated_kp_mask

def construct_interpolated_keypoints2(paths, customized_curves, full_label, interpolated_knots, random_offset=False
                                      ):
    '''
    return a numpy matrix which consists of interpolated keypoints
    :param paths: list, N_path x 2, [[path0_kp1_index, path0_kp2_index], [path1_kp1_index, path1_kp2_index], ...]
    :param customized_curves: numpy, N_path x 3, each row represents three curves (line, clockwise curve, anti-clockwise curve) for one path,
                              1 means being choosed, 0 means not being choosed, for example, [1, 0, 1]
    :param full_label: numpy, all of the annotated kps, numpy, N_fullkps x 3, each row is (x, y, is_visible)
    :param interpolated_knots: numpy, a series of values in [0, 1], for instance, [0.25, 0.5, 0.75]
    :return:
             interpolated_kps, numpy, (N_curves * N_knots) x 2; N_curves = sum(customized_curves)
             interpolated_kps_mask, numpy, (N_curves * N_knots)
    '''
    N_path = len(paths)  # N_path x 2
    N_knots = len(interpolated_knots)  # N_knots
    curves_per_path = customized_curves.shape[1]  # N_path x curves_per_path
    N_curves = int(np.sum(customized_curves))
    N_kps = N_curves * N_knots
    interpolated_kps = np.zeros((N_kps, 2))
    interpolated_kp_mask = np.zeros(N_kps, dtype=np.float32)

    curves_count = 0
    for i in range(N_path):
        (start_kp_ind, end_kp_ind) = paths[i]
        # judge if the start point and end point both exist; existing = 1, not existing = 0
        if full_label[start_kp_ind, 2] == 0 or full_label[end_kp_ind, 2] == 0:
            n_skip_curves = int(np.sum(customized_curves[i, :]))
            curves_count += n_skip_curves
            continue
        start_pt = full_label[start_kp_ind, 0:2]  # (x1, y1)
        end_pt = full_label[end_kp_ind, 0:2]  # (x2, y2)
        for j in range(curves_per_path):  # 0 ~ N_curves - 1
            if customized_curves[i, j] == 0:
                continue
            if j == 0:  # linear interpolation
                w1 = 1 - interpolated_knots
                w2 = interpolated_knots
                pts = start_pt * w1.reshape(N_knots, 1) + end_pt * w2.reshape(N_knots, 1)  # using broadcast, N_knots x 2
            elif j == 1:  # elliptic curve interpolation, clockwise
                # major axis: start_pt --> end_pt
                (x1, y1) = start_pt
                (x2, y2) = end_pt
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                semi_major_axis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2
                semi_minor_axis = semi_major_axis / 4  # can be tuned
                radian = math.atan2(-y2 + y1, x2 - x1)  # (-y2 - (-y1), x2 - x1), [0, pi] U (-pi, 0)
                u = -math.pi * interpolated_knots  # interpolation in clockwise direction
                x = semi_major_axis * np.cos(u)
                y = semi_minor_axis * np.sin(u)
                x_image = np.cos(radian) * x - np.sin(radian) * y + center_x
                y_image = -np.sin(radian) * x - np.cos(radian) * y + center_y
                pts = np.concatenate((x_image.reshape(-1, 1), y_image.reshape(-1, 1)), axis=1)
            elif j == 2:  # elliptic curve interpolation, anti-clockwise
                # major axis: start_pt --> end_pt
                (x1, y1) = start_pt
                (x2, y2) = end_pt
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                semi_major_axis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2
                semi_minor_axis = semi_major_axis / 4  # can be tuned
                radian = math.atan2(-y2 + y1, x2 - x1)  # (-y2 - (-y1), x2 - x1), [0, pi] U (-pi, 0)
                u = math.pi * interpolated_knots  # interpolation in clockwise direction
                x = semi_major_axis * np.cos(u)
                y = semi_minor_axis * np.sin(u)
                x_image = np.cos(radian) * x - np.sin(radian) * y + center_x
                y_image = -np.sin(radian) * x - np.cos(radian) * y + center_y
                pts = np.concatenate((x_image.reshape(-1, 1), y_image.reshape(-1, 1)), axis=1)

            interpolated_kps[(curves_count * N_knots): ((curves_count + 1) * N_knots), :] = pts
            interpolated_kp_mask[(curves_count * N_knots): ((curves_count + 1) * N_knots)] = 1
            curves_count += 1

    # if curves_count != N_curves:
    #     print('n curves:%d'%N_curves)
    #     print('real curves:%d'%curves_count)
    return interpolated_kps, interpolated_kp_mask


def AnnotationPrepare(image_roots, keypoint_annotation_paths, anno_save_root, anno_save_name=None):
    '''
    changing the filename of each sample with its full image path, combing several annotations together and saving
    them in json file

    :param image_roots:
    :param keypoint_annotation_paths:
    :param save_root:
    :return:
    '''
    total_samples = []
    for i in range(len(keypoint_annotation_paths)):
        with open(keypoint_annotation_paths[i], 'r') as fin:
            dataset = json.load(fin)
            subset_samples = dataset['anns']
            fin.close()
        for each_sample in subset_samples:
            filename = each_sample['filename']
            # each_sample['filename'] = os.path.join(image_roots[i], filename)
            each_sample['filename'] = os.path.join(image_roots[i], dataset['classid2name'][each_sample['classid']], filename)
            each_sample['category'] = dataset['classid2name'][each_sample['classid']]
        total_samples += subset_samples

    # remove illegal keypoint annotations and disable it
    for ind in range(len(total_samples)):
        # print(total_samples[ind]['filename'])
        # w, h = Image.open(total_samples[ind]['filename']).size
        w, h = total_samples[ind]['w_h']
        for i, kp_type in enumerate(KEYPOINT_TYPES):
            x, y = total_samples[ind]['part'][i][0], total_samples[ind]['part'][i][1]
            flag = x < 0 or x >= w or y < 0 or y >= h
            if flag:
                total_samples[ind]['part'][i][0] = 0
                total_samples[ind]['part'][i][1] = 0
                total_samples[ind]['part'][i][2] = 0

    if anno_save_name == None:
        _, anno_name = os.path.split(keypoint_annotation_paths[0])
        save_annotation_path = os.path.join(anno_save_root, anno_name)
    else:
        save_annotation_path = os.path.join(anno_save_root, anno_save_name)

    if os.path.exists(anno_save_root) is False:
        os.makedirs(anno_save_root)
    with open(save_annotation_path, 'w') as fout:
        dataset['anns'] = total_samples
        # json.dump(total_samples, fout)
        json.dump(dataset, fout)
        fout.close()

def AnnotationPrepareAndSplit(image_roots, keypoint_annotation_paths, anno_save_root, split_ratio=0.7):
    '''
    changing the filename of each sample with its full image path, combing several annotations together and saving
    them in json file

    :param image_roots:
    :param keypoint_annotation_paths:
    :param save_root:
    :param split_ratio: training vs testing, which will result in two files, for example 'dog_train.json', 'dog_split_test.json'
    :return:
    '''
    total_samples = []
    for i in range(len(keypoint_annotation_paths)):
        with open(keypoint_annotation_paths[i], 'r') as fin:
            dataset = json.load(fin)
            subset_samples = dataset['anns']
            fin.close()
        for each_sample in subset_samples:
            filename = each_sample['filename']
            # each_sample['filename'] = os.path.join(image_roots[i], filename)
            each_sample['filename'] = os.path.join(image_roots[i], dataset['classid2name'][each_sample['classid']], filename)
            each_sample['category'] = dataset['classid2name'][each_sample['classid']]
        total_samples += subset_samples

    # remove illegal keypoint annotations and disable it
    for ind in range(len(total_samples)):
        # print(total_samples[ind]['filename'])
        # w, h = Image.open(total_samples[ind]['filename']).size
        w, h = total_samples[ind]['w_h']
        for i, kp_type in enumerate(KEYPOINT_TYPES):
            x, y = total_samples[ind]['part'][i][0], total_samples[ind]['part'][i][1]
            flag = x < 0 or x >= w or y < 0 or y >= h
            if flag:
                total_samples[ind]['part'][i][0] = 0
                total_samples[ind]['part'][i][1] = 0
                total_samples[ind]['part'][i][2] = 0

    _, anno_name = os.path.split(keypoint_annotation_paths[0])
    anno_name_without_ext, ext = os.path.splitext(anno_name)
    train_annotation_path = os.path.join(anno_save_root, anno_name_without_ext+'_split_train.json')
    test_annotation_path = os.path.join(anno_save_root, anno_name_without_ext + '_split_test.json')

    index_set = set(range(len(total_samples)))
    num_train = round(split_ratio * len(total_samples))
    train_samples_id = random.sample(index_set, num_train)
    test_samples_id = index_set.difference(train_samples_id)
    train_samples = [total_samples[ind] for ind in train_samples_id]
    test_samples = [total_samples[ind] for ind in test_samples_id]
    with open(train_annotation_path, 'w') as fout:
        train_set = {}
        train_set['classid2name'] = dataset['classid2name']
        train_set['anns'] = train_samples
        # json.dump(train_samples, fout)
        json.dump(train_set, fout)
        fout.close()
    with open(test_annotation_path, 'w') as fout:
        test_set = {}
        test_set['classid2name'] = dataset['classid2name']
        test_set['anns'] = test_samples
        # json.dump(test_samples, fout)
        json.dump(test_set, fout)
        fout.close()

class EpisodeGenerator(object):
    def __init__(self, keypoint_annotation_path, N_way, K_shot, M_queries, kp_category_set=None, order_fixed=False, vis_requirement='full_visible', least_support_kps_num=3, least_query_kps_num=3, episode_type='mix_class'):
        '''
        :param keypoint_annotation_path:
        :param N_way: number of classes of support keypoints each support image should have
        :param K_shot: number of support images
        :param M_queries: number of query images
        :param kp_category_set:
        :param order_fixed: False, doing sampling keypoint types every episode; True, fixed order for keypoint types
        :param vis_requirement: 'full_visible' or 'partial_visible'
        :param least_support_kps_num: the least number of union of support keypoint types
        :param least_query_kps_num: the least number of keypoints that can be queried in each query image
        :param episode_type: 'one_class' or 'mix_class', refers to the images in episode from one class or multiple classes, namely one-class episode or mix-class episode

        Class members:
            self.samples

            self.num_category
            self.num_support
            self.num_query
            self.kp_category_set
            self.order_fixed
            self.vis_requirement
            self.least_support_kps_num
            self.least_query_kps_num

            self.supports
            self.queries
        '''
        self.samples = None
        with open(keypoint_annotation_path, 'r') as fin:
            # self.samples = json.load(fin)
            dataset = json.load(fin)
            self.samples = dataset['anns']
            fin.close()
        self.num_category = N_way
        self.num_support = K_shot
        self.num_query = M_queries
        self.kp_category_set = kp_category_set  # pre-defined category set for training or testing
        self.support_kp_categories = None  # support keypoint categories that are picked from keypoint set
        self.num_valid_image = 0
        self.episode_type = episode_type  # 'one_class' or 'mix_class'

        # whether apply sampling strategy for support_kp_categories; if True, not sampling and keypoints' order will be fixed from begin to end
        self.order_fixed = order_fixed
        # 'fully_visible' means samples should have all keypoints visible
        # while 'partial_visible' means the sample could have partial visible kps
        self.vis_requirement = vis_requirement
        # smallest number of union visible kps needed as support
        self.least_support_kps_num = least_support_kps_num
        # smallest number of visible kps per image needed as query
        self.least_query_kps_num = least_query_kps_num
        self.support_kp_mask = torch.ones(K_shot, N_way)
        self.query_kp_mask = torch.ones(M_queries, N_way)
        self.fixed_candidates_ids = None  # used when self.order_fixed = True

        # if training keypoint orders are fixed, it will be pre-defined from begin to end,
        # otherwise, the support_kp_categories will be sampled in .next() function
        if self.order_fixed == True:
            if kp_category_set != None and N_way == len(kp_category_set):
                self.support_kp_categories = kp_category_set
            elif kp_category_set == None and N_way == len(KEYPOINT_TYPES):
                self.support_kp_categories = KEYPOINT_TYPES
            else:
                print('Error in EpisodeGenerator: wrong configuration in fixed-ordered training keypoints!')
                exit(0)
        else:  # self.order_fixed == False, the self.support_kp_categories will be sampled in .next()
            if self.kp_category_set == None:
                self.kp_category_set = KEYPOINT_TYPES


        if self.support_kp_categories != None:
            # N way should <= the number of specified categories
            assert (self.num_category <= len(self.support_kp_categories))

        # build an index map when image i falls in the list of keypoint category j
        # self.imagelist = OrderedDict()
        # nsamples = len(self.samples)
        # for kp_type in KEYPOINT_TYPES:
        #     self.imagelist[kp_type] = []
        # for i in range(nsamples):
        #     each_sample = self.samples[i]
        #     keypoints = each_sample['part']
        #     for j, kp_type in enumerate(KEYPOINT_TYPES):
        #         if keypoints[j][2] != 0: # namely if image i has this keypoint category kp_type
        #             self.imagelist[kp_type].append(i) # append image i into the list

        self.global_imagelist = OrderedDict()
        self.perclass_imagelist = OrderedDict()
        self.perclass_imagenum = OrderedDict()
        nsamples = len(self.samples)
        for kp_type in KEYPOINT_TYPES:
            self.global_imagelist[kp_type] = []
        for i in range(nsamples):
            each_sample = self.samples[i]
            classid = each_sample['classid']
            keypoints = each_sample['part']
            if classid not in self.perclass_imagelist:
                self.perclass_imagelist[classid] = OrderedDict()
                for kp_type in KEYPOINT_TYPES:
                    self.perclass_imagelist[classid][kp_type] = []
                self.perclass_imagenum[classid] = 0  # used for counting
            for j, kp_type in enumerate(KEYPOINT_TYPES):
                if keypoints[j][2] != 0:  # namely if image i has this keypoint category kp_type
                    self.global_imagelist[kp_type].append(i)  # append image i into the list
                    self.perclass_imagelist[classid][kp_type].append(i)
            self.perclass_imagenum[classid] += 1
        self.class_list = list(self.perclass_imagenum.keys())  # a list containing classid

        perclass_imagenum_value = np.array([self.perclass_imagenum[classid] for classid in self.class_list])
        self.perclass_prob = perclass_imagenum_value / np.sum(perclass_imagenum_value)

        if self.order_fixed == True:  # used when self.order_fixed = True
            if self.episode_type == 'mix_class':
                self.fixed_candidates_ids = self.acquire_image_candidates_ids(self.global_imagelist, self.support_kp_categories)  # a list
            else:  # 'one_class'
                self.fixed_candidates_ids_dict = self.acquire_image_candidates_ids2()  # a dict, each value (dictinary[classid]) is a list

        self.supports = None
        self.queries = None

        self.episode_next()  # call to initialize self.num_valid_image

    def acquire_image_candidates_ids(self, imagelist: dict, support_kp_categories: list):
        candidates_ids = set()
        for kp_type in support_kp_categories:
            candidates_ids.update(set(imagelist[kp_type]))
        return list(candidates_ids)

    def acquire_image_candidates_ids2(self):
        candidates_dict = {}
        for classid in self.perclass_imagelist:
            candidates_dict[classid] = set()
            for kp_type in self.support_kp_categories:
                candidates_dict[classid].update(set(self.perclass_imagelist[classid][kp_type]))
        return candidates_dict

    def episode_next(self):
        '''
        generate a new episode which will update self.supports and self.queries
        '''
        # randomly sample N_way keypoint categories from self.kp_category_set
        # dynamic support_kp_categories
        if self.order_fixed == False:
            self.support_kp_categories = random.sample(self.kp_category_set, self.num_category)

        if self.vis_requirement == 'full_visible':
            # find the keypoint type with minimal number of images in self.imagelist
            if self.episode_type == 'mix_class':
                imagelist = self.global_imagelist
            else:  # self.episode_type == 'one_class'
                classid = np.random.choice(self.class_list, size=1, replace=True, p=self.perclass_prob)  # sample a class
                imagelist = self.perclass_imagelist[classid]
            type_with_min_num = self.support_kp_categories[0]
            num_temp = len(imagelist[self.support_kp_categories[0]])
            for kp_type in self.support_kp_categories:
                length = len(imagelist[kp_type])
                if  length <= num_temp:
                    num_temp = length
                    type_with_min_num = kp_type

            # filter and get the candidate samples ids, whose image all has the support keypoint categories
            candidate_samples_ids = []
            for i in imagelist[type_with_min_num]:
                # judge whether image i has all the support categories
                isQualified = True
                each_sample = self.samples[i]
                keypoints = each_sample['part']
                for kp_type in self.support_kp_categories:
                    kp_id = KEYPOINT_TYPE_IDS[kp_type]
                    if keypoints[kp_id][2] == 0:
                        isQualified = False
                        break
                if isQualified == True:
                    candidate_samples_ids.append(i)

            # the sum of the number of desired supports and queries should <= the number of candidates
            if ((self.num_support + self.num_query) > len(candidate_samples_ids)):
                return False

            # sample supports and queries
            self.num_valid_image = len(candidate_samples_ids)
            random.shuffle(candidate_samples_ids)
            self.supports = [self.samples[i] for i in candidate_samples_ids[:self.num_support]]
            self.queries = [self.samples[j] for j in candidate_samples_ids[self.num_support:(self.num_support+self.num_query)]]

        elif self.vis_requirement == 'partial_visible':
            # if self.order_fixed == False, we could update the pool of image candidates (namely self.fixed_candidates_ids) in each iteration
            # dynamic sampling support_kp_categories from kp_category_set and thus update the pool of images
            # t1 = time.time()
            if self.order_fixed == False:
                if self.episode_type == 'mix_class':
                    self.fixed_candidates_ids = self.acquire_image_candidates_ids(self.global_imagelist, self.support_kp_categories)
                else:  # self.episode_type == 'one_class'
                    classid = np.random.choice(self.class_list, size=1, replace=True, p=self.perclass_prob)  # sample a class
                    classid = classid[0]  # get value
                    imagelist = self.perclass_imagelist[classid]
                    self.fixed_candidates_ids = self.acquire_image_candidates_ids(imagelist, self.support_kp_categories)
            else:  # self.order_fixed == True
                if self.episode_type == 'mix_class':
                    pass  # directly use self.fixed_candidates_ids
                else:  # self.episode_type == 'one_class'
                    classid = np.random.choice(self.class_list, size=1, replace=True, p=self.perclass_prob)  # sample a class
                    classid = classid[0]  # get value
                    self.fixed_candidates_ids = self.fixed_candidates_ids_dict[classid]
            # t2 = time.time()
            # print("time count: ", t2-t1)

            # the sum of the number of desired supports and queries should <= the number of candidates
            if ((self.num_support + self.num_query) > len(self.fixed_candidates_ids)):
                return False

            # filter and get the candidate samples ids
            self.num_valid_image = len(self.fixed_candidates_ids)
            candidate_samples_ids = random.sample(self.fixed_candidates_ids, (self.num_support + self.num_query))
            self.supports = [self.samples[i] for i in candidate_samples_ids[:self.num_support]]
            self.queries = [self.samples[j] for j in candidate_samples_ids[self.num_support:(self.num_support + self.num_query)]]

            # compute self.support_kp_mask, torch.Tensor, K_shot x N_way
            for i, each_sample in enumerate(self.supports):
                for j, kp_type in enumerate(self.support_kp_categories):
                    kp_id = KEYPOINT_TYPE_IDS[kp_type]
                    if each_sample['part'][kp_id][2] == 0:  # invisible
                        self.support_kp_mask[i, j] = 0
                    else:
                        self.support_kp_mask[i, j] = 1

            # compute self.query_kp_mask, torch.Tensor, M_queries x N_way
            for i, each_sample in enumerate(self.queries):
                for j, kp_type in enumerate(self.support_kp_categories):
                    kp_id = KEYPOINT_TYPE_IDS[kp_type]
                    if each_sample['part'][kp_id][2] == 0:  # invisible
                        self.query_kp_mask[i, j] = 0
                    else:
                        self.query_kp_mask[i, j] = 1

            # compute the union of keypoint types in sampled images, N(union) <= N_way
            union_support_kp_mask = torch.sum(self.support_kp_mask, dim=0) > 0  # tensor([True, False, True, ...])
            if torch.sum(union_support_kp_mask) < self.least_support_kps_num:
                return False

            # compute the valid query keypoints, using broadcast
            valid_kp_mask = self.query_kp_mask * union_support_kp_mask.reshape(1, self.num_category)
            valid_list = torch.sum(valid_kp_mask, dim=1) >= self.least_query_kps_num  # number of valid keypoints per query
            if torch.sum(valid_list) < valid_list.size(0) * 0.4:  # valid query samples
                return False

        return True

class CUBDataset(Dataset):
    def __init__(self, sample_list, support_kp_categories, using_auxiliary_keypoints=False, interpolation_knots=None, interpolation_mode=3, auxiliary_path=[], hdf5_images_path=None, saliency_maps_root=None, output_saliency_map=False,
                 preprocess=None, input_transform=None, target_transform=None):
        """AnimalPoseDataset

        Arguments:
            image_root {str}
            sample_list {list} -- [{'filename': image_folder/cat1.jpg, 'category': cat, 'keypoints': [{'l_eye': [x, y, isvisible]}, ...], 'visible_bounds': [height, width, xmin, ymin]}]

        Keyword Arguments:
            input_transform {Callable} -- preprocessing for images (default: {None})
            target_transform {Callable} -- preprocessing for labels (default: {None})

        Members:
        """
        self.sample_list = sample_list
        self.keypoint_categories = support_kp_categories
        self.preprocess = preprocess  # callable object
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.using_auxiliary_keypoints = using_auxiliary_keypoints
        if self.using_auxiliary_keypoints == True:
            # define a list containing the paths, each path is represented by kp index pair [index1, index2]
            # our paths are subject to support keypoints; then we will interpolated kps for each path. The paths is possible to be empty list []
            # self.auxiliary_paths = get_auxiliary_paths(path_mode=auxiliary_path_mode, support_keypoint_categories=support_kp_categories)  # 'exhaust', 'predefined'
            self.auxiliary_paths = auxiliary_path
            self.interpolation_knots = interpolation_knots  # np.array([0.25, 0.5, 0.75])  # will result in (N_knots * N_paths) auxiliary keypoints
            self.interpolation_mode=interpolation_mode
        self.saliency_maps_root = saliency_maps_root
        self.T_saliency = int(255 * 0.125)  # for the saliency images generated by SCRN, T=1/8 * 255
        self.output_saliency_map = output_saliency_map

        self.hdf5_images_path = hdf5_images_path
        self.hdf5_images_fin = None  # hdf5 file object for images
        self.hdf5_smaps_fin = None  # hdf5 file object for saliency maps
        if hdf5_images_path is not None:  # read images (or saliency maps) from hdf5 files instead of raw images
            # self.hdf5_images_fin = h5py.File(hdf5_images_path, 'r')
            # # self.hdf5_images_folder_name, _ = os.path.splitext(hdf5_images_path)
            # if self.saliency_maps_root is not None:
            #     self.hdf5_smaps_fin = h5py.File(saliency_maps_root, 'r')

            self.hdf5_images_path = hdf5_images_path

        # self.keypoint_mask = copy.deepcopy(keypoint_mask)
        # self.flip_swap_index = self.get_swap_index()  # when swapping the values of keypoint types, a swap index is needed


    def __getitem__(self, index):
        each_sample = self.sample_list[index]
        # note that the filename is the full path since we use AnnotationPrepare() to process it to be the path for convenience
        image_path = each_sample['filename']
        category = each_sample['category']
        keypoints = each_sample['part']
        visible_bounds = each_sample['bbx']  # [x_min, x_max, y_min, y_max]

        if self.hdf5_images_path is None:  # read from raw files
            image = Image.open(image_path).convert('RGB')
            w, h = image.size  # PIL image
        else:  # read from hdf5
            hdf5_images_fin = h5py.File(self.hdf5_images_path, 'r')
            # key_for_image = image_path.replace(self.hdf5_images_folder_name, "", 1)
            _, filename = os.path.split(image_path)
            key_for_image = os.path.join(category, filename)
            jpeg_stream = hdf5_images_fin[key_for_image]
            image = cv2.imdecode(jpeg_stream[()], cv2.IMREAD_COLOR)  # cv2.IMREAD_UNCHANGED
            image = image[:, :, [2,1,0]]  # rgb
            image = Image.fromarray(image, mode='RGB')
            w, h = image.size
            hdf5_images_fin.close()  # close the hdf5 file
        w_h_origin = np.array([w, h])

        saliency_image = 0
        if (self.saliency_maps_root is not None) and (self.using_auxiliary_keypoints == True or self.output_saliency_map == True):
            # open saliency image, merge it with RGB image in order to perform geometry transformation together
            if self.hdf5_images_path is None:  # read from raw files
                _, filename = os.path.split(image_path)
                filename_wo_ext, _ = os.path.splitext(filename)
                salilency_map_path = os.path.join(self.saliency_maps_root, category, filename_wo_ext + '.png')
                # print(salilency_map_path)
                saliency_image = Image.open(salilency_map_path)  # .convert('L')
            else:  # read from hdf5
                hdf5_smaps_fin = h5py.File(self.saliency_maps_root, 'r')
                _, filename = os.path.split(image_path)
                filename_wo_ext, _ = os.path.splitext(filename)
                key_for_smap = os.path.join(category, filename_wo_ext + '.png')
                # print(key_for_smap)
                jpeg_stream = hdf5_smaps_fin[key_for_smap]
                saliency_image = cv2.imdecode(jpeg_stream[()], cv2.IMREAD_UNCHANGED)
                saliency_image = Image.fromarray(saliency_image)  # 1 channel saliency map
                hdf5_smaps_fin.close()  # close the hdf5 file

            R, G, B = image.split()
            image = Image.merge('RGBA', (R,G,B,saliency_image))
            # image.show()

        # label = [[keypoints[kp_type][0], keypoints[kp_type][1]] for kp_type in self.keypoint_categories]
        # label = np.array(label, np.float)
        # all_labels = [[keypoints[kp_type][0], keypoints[kp_type][1], keypoints[kp_type][2]] for kp_type in KEYPOINT_TYPES]  # (x, y, is_visible)
        all_labels = keypoints  # (x, y, is_visible)
        all_labels = np.array(all_labels, np.float)
        # bbox   = np.array([visible_bounds[2], visible_bounds[3], visible_bounds[1], visible_bounds[0]], np.float)  # [xmin, ymin, w, h]
        bbox = np.array([visible_bounds[0], visible_bounds[2], visible_bounds[1]-visible_bounds[0]+1, visible_bounds[3]-visible_bounds[2]+1]) # [xmin, ymin, w, h]
        # secure our bbox is within the image (some bboxes may be out of boundary)
        bbox[0], bbox[1] = max(bbox[0], 0), max(bbox[1], 0)
        bbox[2], bbox[3] = min(bbox[2], w - bbox[0]), min(bbox[3], h - bbox[1])
        bbox_origin = np.copy(bbox)  # [xmin, ymin, w, h]

        anno = {
            'keypoints': all_labels,
            'bbox': bbox
        }
        meta = {
            'scale': 1.0,
            'offset': np.array([0, 0], np.float),
            'pad_offset': np.array([0, 0], np.float),
            'valid_area': np.array([0, 0, w, h], np.float),
            'hflip': False,  # may randomly flip and set it to be True in preprocess
            'im_index': index,
        }
        # print(all_labels)
        if self.preprocess != None:
            image, anno, meta = self.preprocess(image, anno, meta)
        # re-set those invalid keypoint coordinates since they may be out of boundary
        all_labels_transformed = anno['keypoints']
        for i in range(all_labels_transformed.shape[0]):
            if all_labels_transformed[i, 2] == 0:  # invisible
                all_labels_transformed[i, :2] = 0  # set to be 0

        # scale, xoffset, yoffset, bbx_area, pad_xoffset, pad_yoffset
        scale_trans = np.array([meta['scale'], meta['offset'][0], meta['offset'][1], anno['bbox'][2] * anno['bbox'][3], meta['pad_offset'][0], meta['pad_offset'][1]])

        # extract the transformed keypoint labels relevant to our support keypoints
        label = np.zeros((len(self.keypoint_categories), 2))       # N x 2
        keypoint_mask = torch.ones(len(self.keypoint_categories))  # N
        for i, kp_type in enumerate(self.keypoint_categories):
            kp_id = KEYPOINT_TYPE_IDS[kp_type]
            label[i, :] = all_labels_transformed[kp_id, :2]
            if all_labels_transformed[kp_id, 2] == 0:  # invisible
                keypoint_mask[i] = 0
            else:
                keypoint_mask[i] = 1

        if (self.saliency_maps_root is not None) and (self.using_auxiliary_keypoints == True or self.output_saliency_map == True):
            # separate RGB image and saliency image
            R, G, B, saliency_image = image.split()
            image = Image.merge('RGB', (R, G, B))
            # saliency_image.show()
            # image.show()
            saliency_image = np.array(saliency_image, dtype=np.uint8)

        interpolated_kps, interpolated_kp_mask = 0, 0
        if self.using_auxiliary_keypoints:
            # image.show()
            # saliency_image.show()
            # =============================
            # construct auxiliary keypoints
            # -----#1, linear interpolation-----
            # interpolated_kps: T * 2; interpolated_kp_mask: T; T = (N_knots * N_paths)
            # interpolated_kps, interpolated_kp_mask = construct_interpolated_keypoints(self.auxiliary_paths, all_labels, self.interpolation_knots)

            # -----#2, customized interpolation (line, elliptic curve, using a vector to represent, e.g., [1, 1, 0])-----
            # for each path, we can have multiple interpolated curves. Thus the total keypoint num T = sum(customized_curves) * N_knots
            # here we can maximally construct 3 curves for each pair of body parts, one line, two symmetric curves (clockwise, anti-clockwise)
            # it can be choosed by setting 1 or un-used by setting 0 at each row, e.g., [1, 1, 0].
            N_path = len(self.auxiliary_paths)
            customized_curves = np.ones((N_path, 3))
            if self.interpolation_mode == 1:  # 16 curves
                if category == 'cat' or category == 'horse':
                    # nose, eyes, ears are convex
                    customized_curves[0] = [1, 1, 0]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 0, 1]  # nose --> r_ear
                elif category == 'cow' or category == 'sheep':
                    # nose, eyes, ears are concave
                    customized_curves[0] = [1, 0, 1]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 1, 0]  # nose --> r_ear
                elif category == 'dog':  # the convex and concave could both occur
                    pass
            elif self.interpolation_mode == 2:  # 8 curves
                if category == 'cat' or category == 'horse':
                    # nose, eyes, ears are convex
                    customized_curves[0] = [1, 1, 0]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 0, 1]  # nose --> r_ear
                elif category == 'cow' or category == 'sheep':
                    # nose, eyes, ears are concave
                    customized_curves[0] = [1, 0, 1]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 1, 0]  # nose --> r_ear
                elif category == 'dog':  # the convex and concave could both occur
                    pass
                customized_curves[2:, 1] = 0
                customized_curves[2:, 2] = 0
            elif self.interpolation_mode == 3:  # 6 curves
                customized_curves[:, 1] = 0
                customized_curves[:, 2] = 0
            interpolated_kps, interpolated_kp_mask = construct_interpolated_keypoints2(self.auxiliary_paths, customized_curves, all_labels_transformed, self.interpolation_knots)
            # remove illegal keypoints whose coordinates are out of image
            for i in range(len(interpolated_kp_mask)):
                x, y = interpolated_kps[i]
                if x < -1 or x > 1 or y < -1 or y > 1:  # bounds -1~1 instead of 0~w, 0~h
                    interpolated_kp_mask[i] = 0
                    interpolated_kps[i, :2] = 0
            # using saliency map to filter invalid keypoints
            if self.saliency_maps_root is not None:
                # filtering invalid interpolated keypoints
                w_transformed = saliency_image.shape[1]
                for i, is_visible in enumerate(interpolated_kp_mask):
                    if is_visible:
                        # transform coordinates -1~1 to 0~w' (transformed size)
                        x, y = (interpolated_kps[i] / 2.0 + 0.5) * (w_transformed - 1)
                        x, y = int(x+0.5), int(y+0.5)
                        if saliency_image[y, x] < self.T_saliency:  # not fall into salient region
                            interpolated_kp_mask[i] = 0
                            interpolated_kps[i, :2] = 0

        # if meta['hflip'] == True:  # flip keypoint mask for image index-th
        #     self.keypoint_mask[index] = self.keypoint_mask[index][self.flip_swap_index]

        if self.input_transform is not None:
            image = self.input_transform(image)
            if (self.saliency_maps_root is not None) and self.output_saliency_map:
                # transform saliency image (0~255) to tensor (0~1) for extracting patches. Shape is 1 x H x W, single channel
                saliency_image = torchvision.transforms.functional.to_tensor(saliency_image)
                # saliency_image = saliency_image.repeat(3, 1, 1)  # Shape is 3 x H x W
                # saliency_image = saliency_image.permute(1, 2, 0)#.squeeze()
                # im2 = Image.fromarray((saliency_image.numpy()*255).astype(np.uint8), mode='RGB')
                # im2.show()
            else:
                saliency_image = 0
        if self.target_transform is not None:
            label = self.target_transform(label)
            if self.using_auxiliary_keypoints:
                interpolated_kps = self.target_transform(interpolated_kps)

        return image, label, keypoint_mask, scale_trans, interpolated_kps, interpolated_kp_mask, saliency_image, bbox_origin, w_h_origin

    def __len__(self):
        return len(self.sample_list)

    # def get_swap_index(self):
    #     flip_inds = torch.zeros(len(self.keypoint_categories)).long()
    #     for i, kp_type in enumerate(self.keypoint_categories):
    #         swap_kp_type = HFLIP.get(kp_type)
    #         if swap_kp_type != None:
    #             flip_inds[i] = self.keypoint_categories.index(swap_kp_type)
    #         else:
    #             flip_inds[i] = i
    #
    #     return flip_inds



def sample_keypoints(nkps=20, im_h=384, im_w=384, bbx=None, sample_method='random'):
    '''
    nkps: the keypoints to sample
    im_h, im_w: image height and width
    bbx: if not None, sample nkps within bbx (xmin, ymin, w, h)
    saliency_map: if None, evenly sample nkps (point (x, y), 0<=x<=im_w-1, 0<=y<=im_h-1) from an image;
    sample_method: 'random' or 'regular_grid'

    return grids: (nkps+nkps_bg) x 2, each row is a point (x, y)
    '''
    if bbx is None:
        bbx = (0, 0, im_w, im_h)  # (xmin, ymin, w, h)
    xmin, ymin = bbx[0], bbx[1]
    w, h = bbx[2], bbx[3]
    if sample_method == 'random':
        x = np.random.randint(xmin, xmin+w, nkps)
        y = np.random.randint(ymin, ymin+h, nkps)
    elif sample_method == 'regular_grid':
        yx_slices = regular_grid((h, w), nkps)
        y = np.array(range(h)[yx_slices[0]])
        x = np.array(range(w)[yx_slices[1]])
        y = y + ymin
        x = x + xmin

    x, y = x.reshape(nkps, 1), y.reshape(nkps, 1)
    grids = np.concatenate((x, y), axis=1)

    return grids


def sample_keypoints_unevenly(nkps=20, nkps_bg=0, im_h=384, im_w=384, binary_map=None, sample_method='random'):
    '''
    nkps: the keypoints to sample
    im_h, im_w: image height and width
    binary_map: sample nkps in foreground; and also sample nkps_bg in background
    sample_method: 'random'

    return grids: (nkps+nkps_bg) x 2, each row is a point (x, y);
           kp_fgbg_mask: (nkps+nkps_bg) where 1 stands for foreground while 0 stands for background
    '''
    assert binary_map is not None
    grids, kp_fgbg_mask = 0, 0

    return grids, kp_fgbg_mask

from datasets.dataset_utils import draw_markers
def draw_kps(image_list, kps_list):
    for i in range(len(image_list)):
        image = image_list[i]
        kps = kps_list[i]
        raw = np.array(image, dtype=np.uint8)
        im_tmp = raw[:, :, [2,1,0]]
        image = np.zeros(im_tmp.shape, dtype=np.uint8)
        image[:,:,:] = im_tmp[:,:,:]
        keypoint_dict = {i:kps[i] for i in range(len(kps))}
        new_im = draw_markers(image, keypoint_dict, marker='circle', color=[255,255,255], circle_radius=10, thickness=3)
        cv2.imwrite('view{}.png'.format(str(i)), new_im)

class AugmentDataset(Dataset):
    def __init__(self, sample_list, keypoints_num, hdf5_images_path=None, saliency_maps_root=None, output_saliency_map=False,
                 preprocess_list=None, input_transform=None, target_transform=None):
        """AugmentDataset

        Arguments:
            image_root {str}
            sample_list {list} -- [{'filename': image_folder/cat1.jpg, 'category': cat, 'keypoints': [{'l_eye': [x, y, isvisible]}, ...], 'visible_bounds': [height, width, xmin, ymin]}]

        Keyword Arguments:
            input_transform {Callable} -- preprocessing for images (default: {None})
            target_transform {Callable} -- preprocessing for labels (default: {None})

        Members:
        """
        self.sample_list = sample_list
        self.keypoints_num = keypoints_num
        assert preprocess_list is not None
        self.preprocess_list = preprocess_list  # callable object
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.saliency_maps_root = saliency_maps_root
        self.T_saliency = int(255 * 0.1)  # for the saliency images generated by SCRN, T=1/8 * 255
        self.output_saliency_map = output_saliency_map

        self.hdf5_images_path = hdf5_images_path
        self.hdf5_images_fin = None  # hdf5 file object for images
        self.hdf5_smaps_fin = None   # hdf5 file object for saliency maps

        # self.keypoint_mask = copy.deepcopy(keypoint_mask)
        # self.flip_swap_index = self.get_swap_index()  # when swapping the values of keypoint types, a swap index is needed


    def __getitem__(self, index):
        each_sample = self.sample_list[index]
        # note that the filename is the full path since we use AnnotationPrepare() to process it to be the path for convenience
        image_path = each_sample['filename']
        category = each_sample['category']
        keypoints = each_sample['part']
        visible_bounds = each_sample['bbx']  # [x_min, x_max, y_min, y_max]

        if self.hdf5_images_path is None:  # read from raw files
            image = Image.open(image_path).convert('RGB')
            w, h = image.size  # PIL image
        else:  # read from hdf5
            hdf5_images_fin = h5py.File(self.hdf5_images_path, 'r')
            # key_for_image = image_path.replace(self.hdf5_images_folder_name, "", 1)
            _, filename = os.path.split(image_path)
            key_for_image = os.path.join(category, filename)
            jpeg_stream = hdf5_images_fin[key_for_image]
            image = cv2.imdecode(jpeg_stream[()], cv2.IMREAD_COLOR)  # cv2.IMREAD_UNCHANGED
            image = image[:, :, [2,1,0]]  # rgb
            image = Image.fromarray(image, mode='RGB')
            w, h = image.size
            hdf5_images_fin.close()  # close the hdf5 file
        w_h_origin = np.array([w, h])

        if (self.saliency_maps_root is not None) and (self.output_saliency_map == True):
            # open saliency image, merge it with RGB image in order to perform geometry transformation together
            if self.hdf5_images_path is None:  # read from raw files
                _, filename = os.path.split(image_path)
                filename_wo_ext, _ = os.path.splitext(filename)
                salilency_map_path = os.path.join(self.saliency_maps_root, category, filename_wo_ext + '.png')
                # print(salilency_map_path)
                saliency_image = Image.open(salilency_map_path)  # .convert('L')
            else:  # read from hdf5
                hdf5_smaps_fin = h5py.File(self.saliency_maps_root, 'r')
                _, filename = os.path.split(image_path)
                filename_wo_ext, _ = os.path.splitext(filename)
                key_for_smap = os.path.join(category, filename_wo_ext + '.png')
                # print(key_for_smap)
                jpeg_stream = hdf5_smaps_fin[key_for_smap]
                saliency_image = cv2.imdecode(jpeg_stream[()], cv2.IMREAD_UNCHANGED)
                saliency_image = Image.fromarray(saliency_image)  # 1 channel saliency map
                hdf5_smaps_fin.close()  # close the hdf5 file

            R, G, B = image.split()
            image = Image.merge('RGBA', (R,G,B,saliency_image))
            # image.show()

        # sample keypoints
        bbox = np.array([visible_bounds[0], visible_bounds[2], visible_bounds[1]-visible_bounds[0]+1, visible_bounds[3]-visible_bounds[2]+1]) # [xmin, ymin, w, h]
        # secure our bbox is within the image (some bboxes may be out of boundary)
        bbox[0], bbox[1] = max(bbox[0], 0), max(bbox[1], 0)
        bbox[2], bbox[3] = min(bbox[2], w - bbox[0]), min(bbox[3], h - bbox[1])
        bbox_origin = np.copy(bbox)  # [xmin, ymin, w, h]
        all_labels = sample_keypoints(self.keypoints_num, h, w, bbox, sample_method='random')
        all_labels = np.array(all_labels, np.float)
        # t = np.array(copy.deepcopy(image))
        anno = {
            'keypoints': all_labels,
            'bbox': bbox
        }
        meta = {
            'scale': 1.0,
            'offset': np.array([0, 0], np.float),
            'pad_offset': np.array([0, 0], np.float),
            'valid_area': np.array([0, 0, w, h], np.float),
            'hflip': False,  # may randomly flip and set it to be True in preprocess
            'debug': [],  # use to append debug data
        }
        # print(all_labels)
        image1, anno1, meta1 = self.preprocess_list[0](image, anno, meta)
        image2, anno2, meta2 = self.preprocess_list[1](image, anno, meta)

        # test whether raw image will change when performing preprocess such as rotation, crop, ...
        # t2 = np.array(image)
        # s = np.sum((t - t2) ** 2)
        # print(s)
        # if s != 0:
        #     print('wrong')
        #     exit(0)

        # judge1 = np.logical_and(anno1['keypoints'] >= -1, anno1['keypoints'] <= 1)
        # judge2 = np.logical_and(anno2['keypoints'] >= -1, anno2['keypoints'] <= 1)
        # if np.all(judge1) == False or np.all(judge2) == False:
        #     print(anno1['keypoints'])
        #     print(anno2['keypoints'])
        #     print('wrong for out of boundary')
        #     draw_kps([image, image1, image2], [anno['keypoints'], np.clip((anno1['keypoints']/2+0.5)*384, a_min=None, a_max=383), \
        #                                        np.clip((anno1['keypoints']/2+0.5)*384, a_min=None, a_max=383)])
        #     exit(0)

        # print('ok')
        # return 0

        # re-set those invalid keypoint coordinates since they may be out of boundary
        # (they are close to boundary due to calculating error)
        labels1 = np.clip(anno1['keypoints'], a_min=-1, a_max=1)  # preprocess_list should include CoordinateNormalize
        labels2 = np.clip(anno2['keypoints'], a_min=-1, a_max=1)

        # draw_kps([image, image1, image2], [anno['keypoints'], np.clip((anno1['keypoints']/2+0.5)*384, a_min=None, a_max=383), \
        #                                    np.clip((anno2['keypoints']/2+0.5)*384, a_min=None, a_max=383)])

        # scale, xoffset, yoffset, bbx_area, pad_xoffset, pad_yoffset
        scale_trans1 = np.array([meta1['scale'], meta1['offset'][0], meta1['offset'][1], anno1['bbox'][2] * anno1['bbox'][3], meta1['pad_offset'][0], meta1['pad_offset'][1]])
        scale_trans2 = np.array([meta2['scale'], meta2['offset'][0], meta2['offset'][1], anno2['bbox'][2] * anno2['bbox'][3], meta2['pad_offset'][0], meta2['pad_offset'][1]])

        # 1, foreground; 0, background
        kps_fg_mask1 = torch.ones(self.keypoints_num)  # N
        kps_fg_mask2 = torch.ones(self.keypoints_num)  # N
        saliency_image1, saliency_image2 = 0, 0
        if (self.saliency_maps_root is not None) and (self.output_saliency_map == True):
            # separate RGB image and saliency image
            R1, G1, B1, saliency_image1 = image1.split()
            R2, G2, B2, saliency_image2 = image2.split()
            image1 = Image.merge('RGB', (R1, G1, B1))
            image2 = Image.merge('RGB', (R2, G2, B2))
            # saliency_image1.show()
            # image1.show()
            saliency_image1 = np.array(saliency_image1, dtype=np.uint8)
            saliency_image2 = np.array(saliency_image2, dtype=np.uint8)

            # using saliency map to separate fg/bg keypoints
            w_transformed = saliency_image1.shape[1]
            im_keypoints1 = np.clip(np.floor((labels1 / 2.0 + 0.5) * w_transformed).astype(np.int), a_min=None, a_max=w_transformed-1)
            im_keypoints2 = np.clip(np.floor((labels2 / 2.0 + 0.5) * w_transformed).astype(np.int), a_min=None, a_max=w_transformed-1)
            for i in range(self.keypoints_num):
                # transform coordinates -1~1 to 0~w' (transformed size)
                x, y = im_keypoints1[i, :]
                if saliency_image1[y, x] < self.T_saliency:  # not fall into salient region
                    kps_fg_mask1[i] = 0
                x, y = im_keypoints2[i, :]
                if saliency_image2[y, x] < self.T_saliency:  # not fall into salient region
                    kps_fg_mask2[i] = 0

        if self.input_transform is not None:
            image1 = self.input_transform(image1)
            image2 = self.input_transform(image2)
            if (self.saliency_maps_root is not None) and self.output_saliency_map:
                # transform saliency image (0~255) to tensor (0~1) for extracting patches. Shape is 1 x H x W, single channel
                saliency_image1 = torchvision.transforms.functional.to_tensor(saliency_image1)
                saliency_image2 = torchvision.transforms.functional.to_tensor(saliency_image2)
                # w_transformed = saliency_image1.shape[1]
                # saliency_image1 = saliency_image1.expand(3, w_transformed, w_transformed)  # Shape is 3 x H x W
                # saliency_image2 = saliency_image2.expand(3, w_transformed, w_transformed)  # Shape is 3 x H x W
                # saliency_image1 = saliency_image1.permute(1, 2, 0)#.squeeze()
                # im1 = Image.fromarray((saliency_image1.numpy()*255).astype(np.uint8), mode='RGB')
                # im1.show()

        if self.target_transform is not None:
            labels1 = self.target_transform(labels1)
            labels2 = self.target_transform(labels2)

        return image1, image2, labels1, labels2, kps_fg_mask1, kps_fg_mask2, saliency_image1, saliency_image2, \
               scale_trans1, scale_trans2, bbox_origin, w_h_origin

    def __len__(self):
        return len(self.sample_list)



def save_episode_before_preprocess(episode_generator, episode_num=0, delete_old_files=True, draw_main_kps=True, draw_interpolated_kps=False, interpolation_knots=None, interpolation_mode=3, path_mode='predefined', root_postfix=""):
    print(episode_generator.support_kp_categories)
    print(episode_generator.supports)
    print(episode_generator.queries)

    AnimalPose_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_image_part2'
    VOC_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL-VOC2010-TrainVal/VOCdevkit/VOC2011/JPEGImages'
    save_image_root = './episode_images/before-preprocessed'+root_postfix

    if os.path.exists(save_image_root) == False:
        os.makedirs(save_image_root)
    if os.path.exists(save_image_root + "/" + 'support') == False:
        os.makedirs(save_image_root + "/" + 'support')
    if os.path.exists(save_image_root + "/" + 'query') == False:
        os.makedirs(save_image_root + "/" + 'query')

    # remove old episode images
    if delete_old_files == True:
        for each_file in os.listdir(os.path.join(save_image_root, 'support')):
            os.remove(os.path.join(save_image_root, 'support', each_file))
        for each_file in os.listdir(os.path.join(save_image_root, 'query')):
            os.remove(os.path.join(save_image_root, 'query', each_file))

    # write new episode images
    for sample_i, each_sample in enumerate(episode_generator.supports):
        # image_path = os.path.join(AnimalPose_image_root, each_sample['category'], each_sample['filename'])
        image_path = each_sample['filename']
        category = each_sample['category']
        if draw_main_kps:
            keypoints_dict = OrderedDict(zip(KEYPOINT_TYPES, each_sample['part']))
            keypoints_dict = filter_keypoints(keypoints_dict)
            keypoints_dict2 = {}
            keypoints_dict3 = {}
            for kp_type in keypoints_dict.keys():
                # if kp_type in testing_kp_category_set:  # training kp set: training_kp_category_set;  testing_kp_category_set
                #     keypoints_dict2[kp_type] = keypoints_dict[kp_type]
                if kp_type in episode_generator.support_kp_categories:  # choosed kps
                    keypoints_dict3[kp_type] = keypoints_dict[kp_type]

            # image_out = draw_instance(image_path, keypoints_dict, KEYPOINT_TYPES, hightlight_keypoint_types=episode_generator.support_kp_categories, save_root=None, is_show=False)
            # image_out = draw_instance(image_path, keypoints_dict, KEYPOINT_TYPES, hightlight_keypoint_types=None, save_root=None, is_show=False)

            image = cv2.imread(image_path)
            image_out = draw_skeletons(image, [keypoints_dict3], KEYPOINT_TYPES, marker='circle', circle_radius=10, alpha=0.7)
            # image_out = draw_skeletons(image, [keypoints_dict], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=6, alpha=1)
            # image_out = draw_skeletons(image, [keypoints_dict], KEYPOINT_TYPES, marker='diamond', circle_radius=8, alpha=1)

            image_out2 = draw_markers(image_out, keypoints_dict3, marker='circle',  color=[255,255,255], circle_radius=10, thickness=2)
        else:
            image_out = cv2.imread(image_path)

        if draw_interpolated_kps == True:
            keypoints = each_sample['part']  # kp_type: (x, y, is_visible)
            # all_labels = [[keypoints[kp_type][0], keypoints[kp_type][1], keypoints[kp_type][2]] for kp_type in KEYPOINT_TYPES]  # (x, y, is_visible)
            all_labels = keypoints
            all_labels = np.array(all_labels, np.float)
            auxiliary_paths = get_auxiliary_paths(path_mode=path_mode, support_keypoint_categories=episode_generator.support_kp_categories)
            # -----#1, linear interpolation-----
            # will result in (N_knots * N_paths) auxiliary keypoints
            # interpolated_kps: T x 2; interpolated_kp_mask: T; T = (N_knots * N_paths)
            # interpolated_kps, interpolated_kp_mask = construct_interpolated_keypoints(auxiliary_paths, all_labels, interpolation_knots)

            # -----#2, customized interpolation (line, elliptic curve, using a vector to represent, e.g., [1, 1, 0])-----
            # will result in np.sum(customized_curves) * N_knots auxiliary keypoints
            N_path = len(auxiliary_paths)
            customized_curves = np.ones((N_path, 3))
            if interpolation_mode == 1:
                if category == 'cat' or category == 'horse':
                    # nose, eyes, ears are convex
                    customized_curves[0] = [1, 1, 0]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 0, 1]  # nose --> r_ear
                elif category == 'cow' or category == 'sheep':
                    # nose, eyes, ears are concave
                    customized_curves[0] = [1, 0, 1]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 1, 0]  # nose --> r_ear
                elif category == 'dog':  # the convex and concave could both occur
                    pass
            elif interpolation_mode == 2:
                if category == 'cat' or category == 'horse':
                    # nose, eyes, ears are convex
                    customized_curves[0] = [1, 1, 0]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 0, 1]  # nose --> r_ear
                elif category == 'cow' or category == 'sheep':
                    # nose, eyes, ears are concave
                    customized_curves[0] = [1, 0, 1]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 1, 0]  # nose --> r_ear
                elif category == 'dog':  # the convex and concave could both occur
                    pass
                customized_curves[2:, 1] = 0
                customized_curves[2:, 2] = 0
            elif interpolation_mode == 3:
                customized_curves[:, 1] = 0
                customized_curves[:, 2] = 0
            interpolated_kps, interpolated_kp_mask = construct_interpolated_keypoints2(auxiliary_paths, customized_curves, all_labels, interpolation_knots)

            npimg_cur = np.copy(image_out)
            for j, is_visible in enumerate(interpolated_kp_mask):
                if is_visible == 0:
                    continue
                body_part = interpolated_kps[j, :]
                center = (int(body_part[0]), int(body_part[1]))
                cv2.circle(npimg_cur, center, 4, [0, 0, 255], thickness=-1)
                image_out = cv2.addWeighted(image_out, 0.3, npimg_cur, 0.7, 0)

        # save image
        _, filename = os.path.split(image_path)
        filename_wo_ext, ext = os.path.splitext(filename)
        save_root = save_image_root + '/support'
        # cv2.imwrite(os.path.join(save_root, 'eps'+str(episode_num)+'_s{}_'.format(sample_i)+filename_wo_ext + ext), image_out)
        cv2.imwrite(os.path.join(save_root, 'eps'+str(episode_num)+'_s{}_'.format(sample_i)+filename_wo_ext + 'b' + ext), image_out2)

    for sample_i, each_sample in enumerate(episode_generator.queries):
        # image_path = os.path.join(AnimalPose_image_root, each_sample['category'], each_sample['filename'])
        image_path = each_sample['filename']
        if draw_main_kps:
            keypoints_dict = OrderedDict(zip(KEYPOINT_TYPES, each_sample['part']))
            keypoints_dict = filter_keypoints(keypoints_dict)
            keypoints_dict2 = {}
            keypoints_dict3 = {}
            for kp_type in keypoints_dict.keys():
                # if kp_type in testing_kp_category_set:  # training kp set: training_kp_category_set;  testing_kp_category_set
                #     keypoints_dict2[kp_type] = keypoints_dict[kp_type]
                if kp_type in episode_generator.support_kp_categories:  # choosed kps
                    keypoints_dict3[kp_type] = keypoints_dict[kp_type]

            # image_out = draw_instance(image_path, keypoints_dict, KEYPOINT_TYPES, hightlight_keypoint_types=episode_generator.support_kp_categories, save_root=None, is_show=False)

            image = cv2.imread(image_path)
            image_out = draw_skeletons(image, [keypoints_dict3], KEYPOINT_TYPES, marker='circle', circle_radius=10, alpha=0.7)
            # image_out = draw_skeletons(image, [keypoints_dict], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=6, alpha=1)
            # image_out = draw_skeletons(image, [keypoints_dict], KEYPOINT_TYPES, marker='diamond', circle_radius=8, alpha=1)

            image_out2 = draw_markers(image_out, keypoints_dict3, marker='circle', color=[255,255,255], circle_radius=10, thickness=2)
        else:
            image_out = cv2.imread(image_path)

        if draw_interpolated_kps == True:
            keypoints = each_sample['part']  # kp_type: (x, y, is_visible)
            # all_labels = [[keypoints[kp_type][0], keypoints[kp_type][1], keypoints[kp_type][2]] for kp_type in KEYPOINT_TYPES]  # (x, y, is_visible)
            all_labels = keypoints
            all_labels = np.array(all_labels, np.float)
            auxiliary_paths = get_auxiliary_paths(path_mode=path_mode, support_keypoint_categories=episode_generator.support_kp_categories)
            # -----#1, linear interpolation-----
            # will result in (N_knots * N_paths) auxiliary keypoints
            # interpolated_kps: T x 2; interpolated_kp_mask: T; T = (N_knots * N_paths)
            # interpolated_kps, interpolated_kp_mask = construct_interpolated_keypoints(auxiliary_paths, all_labels, interpolation_knots)

            # -----#2, customized interpolation (line, elliptic curve, using a vector to represent, e.g., [1, 1, 0])-----
            # will result in np.sum(customized_curves) * N_knots auxiliary keypoints
            N_path = len(auxiliary_paths)
            customized_curves = np.ones((N_path, 3))
            if interpolation_mode == 1:
                if category == 'cat' or category == 'horse':
                    # nose, eyes, ears are convex
                    customized_curves[0] = [1, 1, 0]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 0, 1]  # nose --> r_ear
                elif category == 'cow' or category == 'sheep':
                    # nose, eyes, ears are concave
                    customized_curves[0] = [1, 0, 1]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 1, 0]  # nose --> r_ear
                elif category == 'dog':  # the convex and concave could both occur
                    pass
            elif interpolation_mode == 2:
                if category == 'cat' or category == 'horse':
                    # nose, eyes, ears are convex
                    customized_curves[0] = [1, 1, 0]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 0, 1]  # nose --> r_ear
                elif category == 'cow' or category == 'sheep':
                    # nose, eyes, ears are concave
                    customized_curves[0] = [1, 0, 1]  # nose --> l_ear, [linear line, clockwise curve, anti-clockwise]
                    customized_curves[1] = [1, 1, 0]  # nose --> r_ear
                elif category == 'dog':  # the convex and concave could both occur
                    pass
                customized_curves[2:, 1] = 0
                customized_curves[2:, 2] = 0
            elif interpolation_mode == 3:
                customized_curves[:, 1] = 0
                customized_curves[:, 2] = 0
            interpolated_kps, interpolated_kp_mask = construct_interpolated_keypoints2(auxiliary_paths, customized_curves, all_labels, interpolation_knots)

            npimg_cur = np.copy(image_out)
            for j, is_visible in enumerate(interpolated_kp_mask):
                if is_visible == 0:
                    continue
                body_part = interpolated_kps[j, :]
                center = (int(body_part[0]), int(body_part[1]))
                cv2.circle(npimg_cur, center, 4, [0, 0, 255], thickness=-1)
                image_out = cv2.addWeighted(image_out, 0.3, npimg_cur, 0.7, 0)

        # save image
        _, filename = os.path.split(image_path)
        filename_wo_ext, ext = os.path.splitext(filename)
        save_root = save_image_root + '/query'
        # cv2.imwrite(os.path.join(save_root, 'eps'+str(episode_num)+'_q{}_'.format(sample_i)+filename_wo_ext + ext), image_out)
        cv2.imwrite(os.path.join(save_root, 'eps'+str(episode_num)+'_q{}_'.format(sample_i)+filename_wo_ext + 'b' + ext), image_out2)

if __name__=='__main__':
    anno_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt/cat.json'
    episode_generator = EpisodeGenerator(anno_path, N_way=3, K_shot=5, M_queries=15)
    episode_generator.episode_next()
    print(episode_generator.support_kp_categories)
    print(episode_generator.supports)
    print(episode_generator.queries)

    AnimalPose_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images'
    save_image_root = './episode_images'

    # remove old episode images
    # for each_file in os.listdir(os.path.join(save_image_root, 'support')):
    #     os.remove(os.path.join(save_image_root, 'support', each_file))
    # for each_file in os.listdir(os.path.join(save_image_root, 'query')):
    #     os.remove(os.path.join(save_image_root, 'query', each_file))

    # write new episode images
    for each_sample in episode_generator.supports:
        image_path = os.path.join(AnimalPose_image_root, each_sample['category'], each_sample['filename'])
        keypoints_dict = OrderedDict(zip(KEYPOINT_TYPES, each_sample['keypoints']))
        draw_instance(image_path, keypoints_dict, KEYPOINT_TYPES, hightlight_keypoint_types=episode_generator.support_kp_categories, save_root=save_image_root + '/support', is_show=True)

    for each_sample in episode_generator.queries:
        image_path = os.path.join(AnimalPose_image_root, each_sample['category'], each_sample['filename'])
        keypoints_dict = OrderedDict(zip(KEYPOINT_TYPES, each_sample['keypoints']))
        draw_instance(image_path, keypoints_dict, KEYPOINT_TYPES, hightlight_keypoint_types=episode_generator.support_kp_categories, save_root=save_image_root + '/query', is_show=False)



