import os
import random

import torch
import json
import argparse
from PIL import Image
import cv2
import numpy as np
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
from datasets.dataset_utils import CocoColors, draw_skeletons, draw_instance
# import sys
# sys.path.append('..')

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

def cub_json_build(dataset_root, save_root=None):
    '''
        All the data entries are contained in a dict, which has two keys,
            {
            'classid2name': ['bird1', 'bird2', ..., 'bird200'],  # a list containing class names
            'anns': [{'filename': 'xxx', 'classid': int, 'part': [[x, y, is_visible],...], 'bbx': [x_min, x_max, y_min, y_max], 'w_h': [w, h]}, ...]
            }  # list
        Each entry in 'anns' is a dict, where its format is as above.
        The path of each image is dataset_root/images/class_name/filename.
    '''
    origin_path = dataset_root
    dataset_dict = {}
    zero_base_for_id = True  # zero base for classid and imageid; if false, it is one base

    id2path = {}
    with open(os.path.join(origin_path, 'images.txt')) as f:
        lines = f.readlines()
        for line in lines:
            index, path = line.strip().split()
            index = int(index)
            if zero_base_for_id == True:
                index -= 1
            id2path[index] = path

    cat2name = {}
    with open(os.path.join(origin_path, 'classes.txt')) as f:
        lines = f.readlines()
        for line in lines:
            cat, name = line.strip().split()
            cat = int(cat)
            if zero_base_for_id == True:
                cat -= 1
            cat2name[cat] = name

    cat2img = {}
    img2cat = {}
    with open(os.path.join(origin_path, 'image_class_labels.txt')) as f:
        lines = f.readlines()
        for line in lines:
            image_id, class_id = line.strip().split()
            image_id = int(image_id)
            class_id = int(class_id)

            if zero_base_for_id == True:
                image_id -= 1
                class_id -= 1

            if class_id not in cat2img:
                cat2img[class_id] = []
            cat2img[class_id].append(image_id)
            img2cat[image_id] = class_id

    id2bbx = {}
    with open(os.path.join(origin_path, 'bounding_boxes.txt')) as f:
        lines = f.readlines()
        for line in lines:
            index, x, y, width, height = line.strip().split()
            index = int(index)
            if zero_base_for_id == True:
                index -= 1
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
            id2bbx[index] = [x, x+width-1, y, y+height-1] # [x, y, width, height]

    id2part = {}
    with open(os.path.join(origin_path, 'parts', 'part_locs.txt')) as f:
        lines = f.readlines()
        for line in lines:
            index, part_id, x, y, visible = line.strip().split()
            index = int(index)
            if zero_base_for_id == True:
                index -= 1
            x = float(x)
            y = float(y)
            visible = int(visible)
            if index not in id2part:
                id2part[index] = []
            id2part[index].append([x, y, visible])

    if zero_base_for_id:
        class_name_list = [cat2name[i] for i in range(len(cat2name))]  # zero base
    else:
        class_name_list = [cat2name[i+1] for i in range(len(cat2name))]  # one base
    dataset_dict['classid2name'] = class_name_list  # 'classid2name': ['bird1', 'bird2', ...], which is a list
    anns = []
    root = '/home/changsheng/LabDatasets/BirdDataset/CUB-200-2011/CUB_200_2011/'
    for id in id2path.keys():
        each_entry = {}
        classfolder, each_entry['filename'] = os.path.split(id2path[id])
        each_entry['classid'] = img2cat[id]
        each_entry['part'] = id2part[id]  # [[x, y, is_visible], ...]
        each_entry['bbx'] = id2bbx[id]    # [xmin, xmax, ymin, ymax]
        image = Image.open(os.path.join(root, 'images', classfolder, each_entry['filename']))
        w, h = image.size
        image.close()
        each_entry['w_h'] = [w, h]
        anns.append(each_entry)
    dataset_dict['anns'] = anns

    if save_root is not None:
        with open(os.path.join(save_root, 'cub_annotations.json'), 'w') as fout:
            json.dump(dataset_dict, fout)
            fout.close()

    return dataset_dict

def cub_dataset_split(dataset_root, save_root=None):
    '''
        All the data entries are contained in a dict, which has two keys,
            {
            'classid2name': ['bird1', 'bird2', ..., 'bird200'],  # a list containing class names
            'anns': [{'filename': 'xxx', 'classid': int, 'part': [[x, y, is_visible],...], 'bbx': [x_min, x_max, y_min, y_max], 'w_h': [w, h]}, ...]
            }  # list
        Each entry in 'anns' is a dict, where its format is as above.
        The path of each image is dataset_root/images/class_name/filename.
    '''
    with open(os.path.join(root, 'cub_annotations.json'), 'r') as fin:
        dataset_dict = json.load(fin)
        fin.close()

    # build a class_id to image_ids list
    cat2img = {}
    for i in range(len(dataset_dict['anns'])):
        entry = dataset_dict['anns'][i]
        classid = entry['classid']
        if classid not in cat2img:
            cat2img[classid] = []
        cat2img[classid].append(i)

    cat2name = dataset_dict['classid2name']

    support = []
    # val_ref = []
    # val_query = []
    # test_ref = []
    # test_query = []
    val = []
    test = []

    #---------------------
    train2 = []  # for each species category, split 70% for training and 30% for testing
    test2 = []
    split_ratio = 0.7
    # ---------------------

    # support_cat = []
    # val_cat = []
    # test_cat = []
    for i in range(0,200):
        img_list = cat2img[i]
        img_num = len(img_list)
        name = cat2name[i]

        if (i%4 == 1) or (i%4 == 3):  # in order to have same partition with PoseNorm-Fewshot paper
            # support_cat.append(name)
            support.extend(img_list)
        elif i%4 == 0:
            # val_cat.append(name)
            # val_ref.extend(img_list[:img_num//5])
            # val_query.extend(img_list[img_num//5:])
            val.extend(img_list)
        elif i%4 ==2:
            # test_cat.append(name)
            # test_ref.extend(img_list[:img_num//5])
            # test_query.extend(img_list[img_num//5:])
            test.extend(img_list)

        # ---------------------
        # for each species category, split 70% for training and 30% for testing
        img_num_sampled = (int)(split_ratio * img_num + 0.5)
        train2_each_species = random.sample(img_list, img_num_sampled)
        test2_each_species = list(set(img_list).difference(set(train2_each_species)))
        train2.extend(train2_each_species)
        test2.extend(test2_each_species)
        # ---------------------

    # copy the instances from dataset_dict to each subsets
    img_id_lists = [support, val, test, train2, test2]
    # cat_id_lists = [support_cat, val_cat, test_cat]
    subset_lists = []
    for i in range(len(img_id_lists)):
        each_dataset = {}
        # each_dataset['classid2name'] = cat_id_lists[i]
        each_dataset['classid2name'] = dataset_dict['classid2name']
        each_dataset['anns'] = [dataset_dict['anns'][j] for j in img_id_lists[i]]  # copy instances
        # for j in range(len(each_dataset['anns'])):  # update the relative classid in each subsets
        #     classid = each_dataset['anns'][j]['classid']
        #     name = cat2name[classid]
        #     new_classid = each_dataset['classid2name'].index(name)
        #     each_dataset['anns'][j]['classid'] = new_classid
        subset_lists.append(each_dataset)
    subset_support, subset_val, subset_test, subset_train2, subset_test2 = subset_lists
    if save_root is not None:
        with open(os.path.join(save_root, 'cub_split_train.json'), 'w') as fout:
            json.dump(subset_support, fout)
            fout.close()
        with open(os.path.join(save_root, 'cub_split_val.json'), 'w') as fout:
            json.dump(subset_val, fout)
            fout.close()
        with open(os.path.join(save_root, 'cub_split_test.json'), 'w') as fout:
            json.dump(subset_test, fout)
            fout.close()

        # for each species category, split 70% for training and 30% for testing
        with open(os.path.join(save_root, 'cub%.2f.json'%split_ratio), 'w') as fout:
            json.dump(subset_train2, fout)
            fout.close()
        with open(os.path.join(save_root, 'cub%.2f.json'%(1-split_ratio)), 'w') as fout:
            json.dump(subset_test2, fout)
            fout.close()




if __name__ == '__main__':
    root = '/home/changsheng/LabDatasets/BirdDataset/CUB-200-2011/CUB_200_2011/'
    dataset_dict = cub_json_build(root, save_root=root)
    cub_dataset_split(root, save_root=root)
    exit(0)
    # root = '../../annotation_prepare/CUB/'

    # with open(os.path.join(root, 'cub_annotations.json'), 'r') as fin:
    #     dataset_dict = json.load(fin)
    #     fin.close()

    with open(os.path.join(root, 'cub_split_train.json'), 'r') as fin:
        dataset_dict_train = json.load(fin)
        fin.close()

    with open(os.path.join(root, 'cub_split_val.json'), 'r') as fin:
        dataset_dict_val = json.load(fin)
        fin.close()

    with open(os.path.join(root, 'cub_split_test.json'), 'r') as fin:
        dataset_dict_test = json.load(fin)
        fin.close()

    # classes_train = []
    # classes_names = []
    # for i, ann in enumerate(dataset_dict_test['anns']):
    #     cls_id = ann['classid']
    #     classes_train.append(cls_id)
    # classes_train = list(set(classes_train))
    # classes_train.sort()
    # for id in classes_train:
    #     classes_names.append(dataset_dict_test['classid2name'][id])
    # with open('cub_test_classes.txt', 'w') as fid:
    #     for i, name in enumerate(classes_names):
    #         fid.writelines(str(i)+' '+name+'\n')
    # fid.close()

    classid2name = dataset_dict_val['classid2name']
    entry = dataset_dict_val['anns'][1]
    path = os.path.join(root, 'images', classid2name[entry['classid']], entry['filename'])
    # path = entry['filename']
    print(root)
    print(path)
    keypoints_dict = {kp_type: entry['part'][i] for i, kp_type in enumerate(get_keypoints())}
    # one_kp_type = 'breast'
    # keypoints = {one_kp_type: entry['part'][get_keypoints().index(one_kp_type)]}
    KEYPOINT_TYPES = get_keypoints()
    draw_instance(path, keypoints_dict, KEYPOINT_TYPES, limbs=[], visible_bounds=[entry['bbx'][3] - entry['bbx'][2], entry['bbx'][1] - entry['bbx'][0],  entry['bbx'][0],  entry['bbx'][2]], hightlight_keypoint_types=None, save_root='.', is_show=True)
    print('ok')

