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




AnimalPose_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images'
AnimalPose_json_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt'
local_json_root = './annotation_prepare'


if os.path.exists(local_json_root) == False:
    os.makedirs(local_json_root)


# annotation preparing
cat_anno_path = AnimalPose_json_root + '/cat.json'
dog_anno_path = AnimalPose_json_root + '/dog.json'
cow_anno_path = AnimalPose_json_root + '/cow.json'
horse_anno_path = AnimalPose_json_root + '/horse.json'
sheep_anno_path = AnimalPose_json_root + '/sheep.json'
AnnotationPrepare([AnimalPose_image_root+'/cat'], [cat_anno_path], anno_save_root=local_json_root)
AnnotationPrepare([AnimalPose_image_root+'/dog'], [dog_anno_path], anno_save_root=local_json_root)
AnnotationPrepare([AnimalPose_image_root+'/cow'], [cow_anno_path], anno_save_root=local_json_root)
AnnotationPrepare([AnimalPose_image_root+'/horse'], [horse_anno_path], anno_save_root=local_json_root)
AnnotationPrepare([AnimalPose_image_root+'/sheep'], [sheep_anno_path], anno_save_root=local_json_root)
# AnnotationPrepareAndSplit([AnimalPose_image_root+'/cat'], [cat_anno_path], anno_save_root=local_json_root, split_ratio=0.7)
# AnnotationPrepareAndSplit([AnimalPose_image_root+'/dog'], [dog_anno_path], anno_save_root=local_json_root, split_ratio=0.7)
exit(0)
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