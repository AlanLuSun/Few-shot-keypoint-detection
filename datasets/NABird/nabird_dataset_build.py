import os
import torch
import json
import argparse
from PIL import Image
import cv2
import numpy as np
import random
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
from datasets.dataset_utils import CocoColors, draw_skeletons, draw_instance
# import sys
# sys.path.append('..')

# 11 keypoints for NABird dataset
def get_keypoints():
    keypoints = [
        'bill',  # namely beak
        'crown',
        'nape',
        'left_eye',
        'right_eye',
        'belly',
        'breast',
        'back',
        'tail',
        'left_wing',
        'right_wing',
    ]
    return keypoints

def nabird_json_build(dataset_root, save_root=None):
    '''
        All the data entries are contained in a dict, which has four keys,
            {
            'classid2name': {classid0: 'bird0', classid1: 'bird1', ...},  # a dict containing pairs of class_id: class_name, all class_id is child class, and this dict is subset of fullclassid2name
            'fullclassid2name': {classid0: 'bird0', classid1: 'bird1', ...}, # a dict containing both child classes and parent classes
            'hierarchy': {child_class_id: parent_class_id},  # used to retrieve each class's parent
            'anns': [{'filename': 'classfolder/filename.jpg', 'classid': int, 'part': [[x, y, is_visible],...], 'bbx': [x_min, x_max, y_min, y_max], 'w_h': [w, h]}, ...]
            }  # list
        Each entry in 'anns' is a dict, where its format is as above.
        The path of each image is dataset_root/images/classid_folder/filename.
    '''
    origin_path = dataset_root
    dataset_dict = {}

    name2size = {}
    with open(os.path.join(origin_path, 'sizes.txt')) as f:
        while True:
            content = f.readline().strip()
            if content == '':
                break
            content = content.split()
            name = content[0].replace('-', '')
            width = int(content[1])
            height = int(content[2])
            name2size[name] = [width, height]

    name2bbx = {}
    with open(os.path.join(origin_path, 'bounding_boxes.txt')) as f:
        while True:
            content = f.readline().strip()
            if content == '':
                break
            content = content.split()
            name = content[0].replace('-', '')
            x = int(content[1])
            y = int(content[2])
            width = int(content[3])
            height = int(content[4])

            x_min = x
            x_max = (x + width - 1)
            y_min = y
            y_max = (y + height - 1)
            name2bbx[name] = [x_min, x_max, y_min, y_max]

    name2part = {}
    with open(os.path.join(origin_path, 'parts/part_locs.txt')) as f:
        while True:
            content = f.readline().strip()
            if content == '':
                break
            content = content.split()
            name = content[0].replace('-', '')
            x = int(content[2])
            y = int(content[3])
            visible = int(content[4])

            if name not in name2part:
                name2part[name] = []
            name2part[name].append([x, y, visible])

    name2path = {}
    with open(os.path.join(origin_path, 'images.txt')) as f:
        lines = f.readlines()
        for line in lines:
            name, path = line.strip().split()
            name = name.replace('-', '')
            name2path[name] = path

    classid2classname = {}
    with open(os.path.join(origin_path, 'classes.txt')) as f:
        lines = f.readlines()
        for line in lines:
            cat_id, name = line.strip().split(" ", maxsplit=1)
            cat_id = int(cat_id)

            classid2classname[cat_id] = name

    name2classid = {}
    child_class_id_list = []
    with open(os.path.join(origin_path, 'image_class_labels.txt')) as f:
        lines = f.readlines()
        for line in lines:
            image_name, class_id = line.strip().split()
            image_name = image_name.replace('-', '')
            class_id = int(class_id)
            name2classid[image_name] = class_id
            child_class_id_list.append(class_id)
    child_class_id_list = list(set(child_class_id_list))  # remove replicate class ids
    child_class_id_list = np.sort(np.array(child_class_id_list))
    childclass2classname = {}
    for child_class_id in child_class_id_list:
        id = int(child_class_id)
        childclass2classname[id] = classid2classname[id]

    class_hierarchy = {}  # 'child_class_id': parent_class_id
    with open(os.path.join(origin_path, 'hierarchy.txt')) as f:
        lines = f.readlines()
        for line in lines:
            child_id, parent_id = line.strip().split()
            child_id, parent_id = int(child_id), int(parent_id)
            class_hierarchy[child_id] = parent_id

    # 'classid2name': [0: 'bird1', 1: 'bird2', ...], which is a dict,
    # and it is also a sub-dict of classid2classname that contains both child class and parent class
    dataset_dict['classid2name'] = childclass2classname  # only contains child class and its class name
    dataset_dict['fullclassid2name'] = classid2classname   # contains both child class and parent class
    dataset_dict['hierarchy'] = class_hierarchy  # 'child_class_id': parent_class_id, used to retrieve each class's parent
    anns = []
    for name in name2classid.keys():
        each_entry = {}
        each_entry['filename'] = name2path[name]
        each_entry['classid'] = name2classid[name]
        each_entry['part'] = name2part[name]  # [[x, y, is_visible], ...]
        each_entry['bbx'] = name2bbx[name]    # [xmin, xmax, ymin, ymax]
        w, h = name2size[name]
        each_entry['w_h'] = [w, h]
        anns.append(each_entry)
    dataset_dict['anns'] = anns

    if save_root is not None:
        with open(os.path.join(save_root, 'nabird_annotations.json'), 'w') as fout:
            json.dump(dataset_dict, fout)
            fout.close()

    return dataset_dict

def nabird_dataset_split(dataset_root, save_root=None):
    '''
        All the data entries are contained in a dict, which has four keys,
            {
            'classid2name': {classid0: 'bird0', classid1: 'bird1', ...},  # a dict containing pairs of class_id: class_name, all class_id is child class, and this dict is subset of fullclassid2name
            'fullclassid2name': {classid0: 'bird0', classid1: 'bird1', ...}, # a dict containing both child classes and parent classes
            'hierarchy': {child_class_id: parent_class_id},  # used to retrieve each class's parent
            'anns': [{'filename': 'classfolder/filename.jpg', 'classid': int, 'part': [[x, y, is_visible],...], 'bbx': [x_min, x_max, y_min, y_max], 'w_h': [w, h]}, ...]
            }  # list
        Each entry in 'anns' is a dict, where its format is as above.
        The path of each image is dataset_root/images/classid_folder/filename.
    '''
    with open(os.path.join(root, 'nabird_annotations.json'), 'r') as fin:
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
    classid_list = list(cat2img.keys())
    classid_list.sort()
    for i in range(0,555):
        cls_id = classid_list[i]
        img_list = cat2img[cls_id]
        img_num = len(img_list)
        name = cat2name[str(cls_id)]

        if (i%5 == 0) or (i%5 == 1) or (i%5 == 2):  # in order to have same partition with PoseNorm-Fewshot paper
            # support_cat.append(name)
            support.extend(img_list)
        elif (i%5 == 3):
            # val_cat.append(name)
            # val_ref.extend(img_list[:img_num//5])
            # val_query.extend(img_list[img_num//5:])
            val.extend(img_list)
        elif (i%5 == 4):
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
        each_dataset['fullclassid2name'] = dataset_dict['fullclassid2name']
        each_dataset['hierarchy'] = dataset_dict['hierarchy']
        each_dataset['anns'] = [dataset_dict['anns'][j] for j in img_id_lists[i]]  # copy instances
        # for j in range(len(each_dataset['anns'])):  # update the relative classid in each subsets
        #     classid = each_dataset['anns'][j]['classid']
        #     name = cat2name[classid]
        #     new_classid = each_dataset['classid2name'].index(name)
        #     each_dataset['anns'][j]['classid'] = new_classid
        subset_lists.append(each_dataset)
    subset_support, subset_val, subset_test, subset_train2, subset_test2 = subset_lists
    if save_root is not None:
        with open(os.path.join(save_root, 'nabird_split_train.json'), 'w') as fout:
            json.dump(subset_support, fout)
            fout.close()
        with open(os.path.join(save_root, 'nabird_split_val.json'), 'w') as fout:
            json.dump(subset_val, fout)
            fout.close()
        with open(os.path.join(save_root, 'nabird_split_test.json'), 'w') as fout:
            json.dump(subset_test, fout)
            fout.close()

        # for each species category, split 70% for training and 30% for testing
        with open(os.path.join(save_root, 'nabird%.2f.json'%split_ratio), 'w') as fout:
            json.dump(subset_train2, fout)
            fout.close()
        with open(os.path.join(save_root, 'nabird%.2f.json'%(1-split_ratio)), 'w') as fout:
            json.dump(subset_test2, fout)
            fout.close()

if __name__ == '__main__':
    root = '/home/changsheng/LabDatasets/BirdDataset/NABird/nabirds/'
    # dataset_dict = nabird_json_build(root, save_root=root)
    # nabird_dataset_split(root, save_root=root)
    # exit(0)
    # root = '../../annotation_prepare/CUB/'

    with open(os.path.join(root, 'nabird_annotations.json'), 'r') as fin:
        dataset_dict = json.load(fin)
        fin.close()

    with open(os.path.join(root, 'nabird_split_train.json'), 'r') as fin:
        dataset_dict_train = json.load(fin)
        fin.close()

    with open(os.path.join(root, 'nabird_split_val.json'), 'r') as fin:
        dataset_dict_val = json.load(fin)
        fin.close()

    with open(os.path.join(root, 'nabird_split_test.json'), 'r') as fin:
        dataset_dict_test = json.load(fin)
        fin.close()

    sets = ['train', 'val', 'test']
    for subset in sets:
        classes_ids = []
        classes_names = []
        dataset_tmp = eval('dataset_dict_'+subset)
        for i, ann in enumerate(dataset_tmp['anns']):
            cls_id = ann['classid']
            classes_ids.append(cls_id)
        classes_ids = list(set(classes_ids))
        classes_ids.sort()
        for id in classes_ids:
            classes_names.append(dataset_tmp['classid2name'][str(id)])
        with open('nabird_%s_classes.txt'%subset, 'w') as fid:
            for i, name in enumerate(classes_names):
                fid.writelines(str(i)+' '+name+'\n')
        fid.close()

    classid2name = dataset_dict_val['classid2name']
    entry = dataset_dict_val['anns'][1]
    path = os.path.join(root, 'images', entry['filename'])
    # path = entry['filename']
    print(root)
    print(path)
    keypoints_dict = {kp_type: entry['part'][i] for i, kp_type in enumerate(get_keypoints())}
    # one_kp_type = 'breast'
    # keypoints = {one_kp_type: entry['part'][get_keypoints().index(one_kp_type)]}
    KEYPOINT_TYPES = get_keypoints()
    draw_instance(path, keypoints_dict, KEYPOINT_TYPES, limbs=[], visible_bounds=[entry['bbx'][3] - entry['bbx'][2], entry['bbx'][1] - entry['bbx'][0],  entry['bbx'][0],  entry['bbx'][2]], hightlight_keypoint_types=None, save_root='.', is_show=True)
    print('ok')

