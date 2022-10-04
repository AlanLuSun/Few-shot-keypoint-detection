import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # accelerate inference when model is not dynamic
# torch.autograd.set_detect_anomaly(True)

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
# import pdb
import json
import time
import datetime
import os
import cv2
import argparse
import logging

# import sys
# sys.path.append('.')  # append pwd into system path so that it could find python modules
# print(os.getcwd())

import datasets.transforms as mytransforms
from datasets.AnimalPoseDataset.animalpose_dataset import AnnotationPrepare, AnnotationPrepareAndSplit, EpisodeGenerator, AnimalPoseDataset, save_episode_before_preprocess
from datasets.dataset_utils import draw_skeletons

from solver_gridms_multiple_kps_covar2 import FSLKeypointNet

############################################################################################
## main call
##
############################################################################################

# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# print(torch.cuda.device_count())

# parser = argparse.ArgumentParser(description='Keypoint detection with few-shot learning')
# parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes')
# parser.add_argument('--use_fused_attention', type=str, default=None)
# args = parser.parse_args()
# print(args.num_episodes)
# print(args.use_fused_attention == None)
# exit(0)


config_str39 = 'image384-2048x12x12(f32)-modu2-gaussian2-x14-reg-adam-80000epochs-11way-dynamicorder-1s5q-all2dog-OneClassEpisode-adapt0-gridMSEx8-12-16-freeze6-0.5vs0.5-3kps6curves-saliency-triplet-covar-wsqa-N11L3S11'

opts = {
    'num_episodes': 80000,
    'N_way': 11,
    'K_shot': 1,
    'M_query': 5,
    'square_image_length': 384,  # 384, 368, 256, 192
    'delete_old_files': False,
    'save_model': False,
    'save_model_root': './savemodel',
    'save_model_postfix': '-%s.pt'%config_str39,
    'load_trained_model': False,
    'load_model_root': './savemodel',   # './savemodel',  '../MyKeypointDetectionV2-NCI/savemodel'
    'load_model_postfix': '-%s.pt'%config_str39,
    'set_eval': False,
    'finetuning_steps': 0,
    'sigma': 14,    # 13, 14
    'eval_method': 'method1',  # 'method1' or 'method2'
    'layer_to_freezing': 6,  # 6, -1
    'downsize_factor': 32,  # 32 (12x12), 46 (8x8), 64 (6x6)
    'grid_length':  [8, 12, 16],  # [8, 12, 16], [16, 18, 20]
    'use_fused_attention': None,  # 'L2-c', 'attmap', None
    'use_domain_confusion': False,
    'use_auxiliary_regressor': False,
    'aux_grid_length':  [8, 12, 16],  # [8, 12, 16], [16, 18, 20]
    'use_interpolated_kps': True,
    'loss_weight': [0.5, 0.5, 0.5],  # [0.5, 0.5, 0.5]
    'interpolation_mode': 3,
    'interpolation_knots': np.array([0.25, 0.5, 0.75]),  # [0.25, 0.5, 0.75], [0.25, 0.375, 0.5, 0.625, 0.75], [0.5]
    'auxiliary_path_mode': 'predefined',  # 'predefined', 'exhaust', 'random'
    'num_random_paths': 6,            # only used when auxiliary_path_mode='random'
    'hdf5_images_path': None,
    'saliency_maps_root': '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/saliency_maps/Animal_Dataset_Combined',
    'sample_times': 15,  # 15, 30
    'eval_compute_var': 0,  # 0, don't compute var in eval; 1, uncorrelated var; 2, covar
    'use_pum': True,  # patches uncertainty module
    'offset_learning': False,  # offset learning net; only used for auxiliary kps
    'data_parallel': False,
    'use_body_part_protos': False,  # used for building universal body part prototypes
    'load_proto': False,
    'memorize_fibers': True,
    'proto_compute_method': 'm',  # 'm': mean; 'ws': weighted_sum
    'eval_with_body_part_protos': False,  # only used in eval
}


if opts['delete_old_files']:
    if os.path.exists('training.log'):
        os.remove('training.log')
    if os.path.exists('runs'):
        import shutil
        shutil.rmtree('runs')

# logging.basicConfig(
#         level=logging.INFO,
#         format='%(message)s',
#         filename='training-%s.log'%config_str3,
#         filemode='a'
#     )
# logging.info('-----------time: {}-----------'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
logging = None
writer_path = './runs/{}-{}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), config_str39)
# writer = SummaryWriter(writer_path)
writer = None

# AnimalPose_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images'
# AnimalPose_json_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt'
local_json_root = './annotation_prepare'

# annotation preparing
# cat_anno_path = AnimalPose_json_root + '/cat.json'
# dog_anno_path = AnimalPose_json_root + '/dog.json'
# cow_anno_path = AnimalPose_json_root + '/cow.json'
# horse_anno_path = AnimalPose_json_root + '/horse.json'
# sheep_anno_path = AnimalPose_json_root + '/sheep.json'
# AnnotationPrepare([AnimalPose_image_root+'/cat'], [cat_anno_path], anno_save_root=local_json_root)
# AnnotationPrepare([AnimalPose_image_root+'/dog'], [dog_anno_path], anno_save_root=local_json_root)
# AnnotationPrepare([AnimalPose_image_root+'/cow'], [cow_anno_path], anno_save_root=local_json_root)
# AnnotationPrepare([AnimalPose_image_root+'/horse'], [horse_anno_path], anno_save_root=local_json_root)
# AnnotationPrepare([AnimalPose_image_root+'/sheep'], [sheep_anno_path], anno_save_root=local_json_root)
# AnnotationPrepareAndSplit([AnimalPose_image_root+'/cat'], [cat_anno_path], anno_save_root=local_json_root, split_ratio=0.7)
# AnnotationPrepareAndSplit([AnimalPose_image_root+'/dog'], [dog_anno_path], anno_save_root=local_json_root, split_ratio=0.7)
# exit(0)
# image_roots = [AnimalPose_image_root+'/cat',
#                # AnimalPose_image_root+'/dog',
#                AnimalPose_image_root+'/cow',
#                AnimalPose_image_root+'/horse',
#                AnimalPose_image_root+'/sheep'
#                ]
# annotation_paths = [cat_anno_path,
#                     # dog_anno_path,
#                     cow_anno_path,
#                     horse_anno_path,
#                     sheep_anno_path
#                     ]
# AnnotationPrepare(image_roots, annotation_paths, anno_save_root='./annotation_prepare', anno_save_name='all_wo_dog.json')
# image_roots = [# AnimalPose_image_root+'/cat',
#                AnimalPose_image_root+'/dog',
#                AnimalPose_image_root+'/cow',
#                AnimalPose_image_root+'/horse',
#                AnimalPose_image_root+'/sheep'
#                ]
# annotation_paths = [# cat_anno_path,
#                     dog_anno_path,
#                     cow_anno_path,
#                     horse_anno_path,
#                     sheep_anno_path
#                     ]
# AnnotationPrepare(image_roots, annotation_paths, anno_save_root='./annotation_prepare', anno_save_name='all_wo_cat.json')




training_kp_category_set = [
        # 'l_eye',
        # 'r_eye',
        'l_ear',
        'r_ear',
        'nose',
        # 'throat',
        # 'withers',
        # 'tail',
        'l_f_leg',
        'r_f_leg',
        'l_b_leg',
        'r_b_leg',
        # 'l_f_knee',
        # 'r_f_knee',
        # 'l_b_knee',
        # 'r_b_knee',
        'l_f_paw',
        'r_f_paw',
        'l_b_paw',
        'r_b_paw'
]


testing_kp_category_set = [
        'l_eye',
        'r_eye',
        # 'l_ear',
        # 'r_ear',
        # 'nose',
        # 'throat',
        # 'withers',
        # 'tail',
        # 'l_f_leg',
        # 'r_f_leg',
        # 'l_b_leg',
        # 'r_b_leg',
        'l_f_knee',
        'r_f_knee',
        'l_b_knee',
        'r_b_knee',
        # 'l_f_paw',
        # 'r_f_paw',
        # 'l_b_paw',
        # 'r_b_paw'
]

s_kp_num = 3  # 3 or 2 or 4 or above
q_kp_num = 3  # 3 or 2 or 4 or above
order_fixed = True  # True
episode_type = "mix_class"  # "one_class", "mix_class"

#----------------- For seen kps -----------------
# N_way1 = len(training_kp_category_set)
if opts['N_way'] > len(training_kp_category_set):
    N_way1 = len(training_kp_category_set)
else:
    N_way1 = opts['N_way']
episode_generator = EpisodeGenerator(os.path.join(local_json_root, 'cat.json'), N_way=N_way1, K_shot=opts['K_shot'], M_queries=opts['M_query'],
                                     kp_category_set=training_kp_category_set, order_fixed=order_fixed, vis_requirement='partial_visible', least_support_kps_num=s_kp_num, least_query_kps_num=q_kp_num, episode_type=episode_type)  # partial_visible  full_visible
print('Number of training images: {} / valid: {}'.format(len(episode_generator.samples), episode_generator.num_valid_image))
episode_generator2 = EpisodeGenerator(os.path.join(local_json_root, 'cow.json'), N_way=N_way1, K_shot=opts['K_shot'], M_queries=opts['M_query'],
                                     kp_category_set=training_kp_category_set, order_fixed=order_fixed, vis_requirement='partial_visible', least_support_kps_num=s_kp_num, least_query_kps_num=q_kp_num, episode_type=episode_type)  # partial_visible  full_visible
print('Number of training images: {} / valid: {}'.format(len(episode_generator2.samples), episode_generator2.num_valid_image))
episode_generator3 = EpisodeGenerator(os.path.join(local_json_root, 'horse.json'), N_way=N_way1, K_shot=opts['K_shot'], M_queries=opts['M_query'],
                                     kp_category_set=training_kp_category_set, order_fixed=order_fixed, vis_requirement='partial_visible', least_support_kps_num=s_kp_num, least_query_kps_num=q_kp_num, episode_type=episode_type)  # partial_visible  full_visible
print('Number of training images: {} / valid: {}'.format(len(episode_generator3.samples), episode_generator3.num_valid_image))
episode_generator4 = EpisodeGenerator(os.path.join(local_json_root, 'sheep.json'), N_way=N_way1, K_shot=opts['K_shot'], M_queries=opts['M_query'],
                                     kp_category_set=training_kp_category_set, order_fixed=order_fixed, vis_requirement='partial_visible', least_support_kps_num=s_kp_num, least_query_kps_num=q_kp_num, episode_type=episode_type)  # partial_visible  full_visible
print('Number of training images: {} / valid: {}'.format(len(episode_generator4.samples), episode_generator4.num_valid_image))

episode_generator_test = EpisodeGenerator(os.path.join(local_json_root, 'dog.json'), N_way=N_way1, K_shot=opts['K_shot'], M_queries=opts['M_query'],
                                     kp_category_set=training_kp_category_set, order_fixed=order_fixed, vis_requirement='partial_visible', least_support_kps_num=s_kp_num, least_query_kps_num=q_kp_num, episode_type=episode_type)
print('Number of testing images: {} / valid: {}'.format(len(episode_generator_test.samples), episode_generator_test.num_valid_image))

#----------------- For unseen kps -----------------
N_way2 = len(testing_kp_category_set)
episode_generator_test2 = EpisodeGenerator(os.path.join(local_json_root, 'dog.json'), N_way=N_way2, K_shot=opts['K_shot'], M_queries=opts['M_query'],
                                     kp_category_set=testing_kp_category_set, order_fixed=order_fixed, vis_requirement='partial_visible', least_support_kps_num=s_kp_num, least_query_kps_num=q_kp_num, episode_type=episode_type)
print('Number of testing images: {} / valid: {}'.format(len(episode_generator_test2.samples), episode_generator_test2.num_valid_image))
# episode_generator_test2 = None

keypointnet = FSLKeypointNet(None, opts, logging, writer, episode_generator_test, episode_generator_test2)
# keypointnet.train()  # training by using single episode generator
keypointnet.train(episode_generator, episode_generator2, episode_generator3, episode_generator4)  # using the multiple


print('******final tests******')
keypointnet.opts['load_trained_model'] = True
keypointnet.opts['load_model_postfix'] = keypointnet.opts['save_model_postfix']
keypointnet.load_model()
print('==============================Test1-seen kps======================================')
keypointnet.validate(episode_generator,  200, eval_method=opts['eval_method'])
# keypointnet.validate(episode_generator2, 100, eval_method=opts['eval_method'])
# keypointnet.validate(episode_generator3, 100, eval_method=opts['eval_method'])
# keypointnet.validate(episode_generator4, 100, eval_method=opts['eval_method'])
print('==============================Test2-seen kps======================================')
keypointnet.validate(episode_generator_test, 1000, eval_method=opts['eval_method'])
# keypointnet.validate(episode_generator_test, 100, eval_method=opts['eval_method'], using_crop=False)
# keypointnet.validate(episode_generator_test, 100, eval_method='method2')
# keypointnet.validate(episode_generator_test, 100, eval_method='method2', using_crop=False)
print('==============================Test3-unseen kps====================================')
keypointnet.validate(episode_generator_test2, 1000, eval_method=opts['eval_method'])
# keypointnet.validate(episode_generator_test2, 100, eval_method=opts['eval_method'], using_crop=False)
# keypointnet.validate(episode_generator_test2, 100, eval_method='method2')
# keypointnet.validate(episode_generator_test2, 100, eval_method='method2', using_crop=False)

