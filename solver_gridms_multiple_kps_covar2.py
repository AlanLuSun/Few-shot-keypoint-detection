import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os
import cv2
import copy

import logging

import datasets.dataset_utils
import datasets.transforms as mytransforms
from datasets.AnimalPoseDataset.animalpose_dataset import EpisodeGenerator, AnimalPoseDataset, KEYPOINT_TYPE_IDS, KEYPOINT_TYPES, save_episode_before_preprocess, kp_connections
from datasets.AnimalPoseDataset.animalpose_dataset import horizontal_swap_keypoints, get_symmetry_keypoints, HFLIP, FBFLIP, DFLIP, get_auxiliary_paths
from datasets.dataset_utils import draw_skeletons, draw_instance, draw_markers


from network.models_gridms2 import Encoder, EncoderMultiScaleType1, EncoderMultiScaleType2
from network.models_gridms2 import feature_modulator2, extract_representations, average_representations, average_representations2, feature_modulator3
from network.models_gridms2 import DescriptorNet
from network.models_gridms2 import RegressorVanila, GridBasedLocator, GridBasedLocator2, GridBasedLocator3, GridBasedLocator4, GridBasedLocatorX, GridBasedLocatorX2, GridBasedLocatorX3, GridBasedLocatorXCovar, GridBasedLocatorX2Covar
from network.models_gridms2 import CovarNet, OffsetLearningNet
from network.models_gridms2 import PatchUncertaintyModule
from network.models_gridms2 import ClassifierGRL
from network.functions import ReverseLayerF
from loss import masked_mse_loss, masked_l1_loss, masked_nll_gaussian, masked_nll_laplacian, instance_weighted_nllloss, instance_weighted_nlllossV0, masked_nll_gaussian_covar, masked_nll_gaussian_covar2

from coco_eval_funs import compute_recall_ap

from utils import print_weights, image_normalize, make_grid_images, make_uncertainty_map, compute_eigenvalues, mean_confidence_interval, mean_confidence_interval_multiple

# torch.autograd.set_detect_anomaly(True)
# import pdb

class FSLKeypointNet(object):
    def __init__(self, episode_generator: EpisodeGenerator, opts: dict, logging = None, writer: SummaryWriter = None,
                 *episode_generator_test: EpisodeGenerator):
        self.episode_generator = episode_generator
        self.opts = opts
        self.logging = logging
        self.writer = writer
        if len(episode_generator_test) == 2:
            self.episode_generator_test = episode_generator_test[0]
            self.episode_generator_test2 = episode_generator_test[1]
        elif len(episode_generator_test) == 1:
            self.episode_generator_test = episode_generator_test[0]
            self.episode_generator_test2 = None
        else:
            print('error for the input of episode_generator_test!')
            exit(0)

        # B x 512 x 46 x 46  (1/8)
        # B x 1024 x 23 x 23 (1/16)
        # B x 2048 x 12 x 12 (1/32)
        # B x 3072 x 8 x 8   (1/46)
        # B x 4096 x 6 x 6   (1/64)
        self.encoder_type = 0  # 0, Encoder, EncoderMultiScaleType1; 1, EncoderMultiScaleType2
        self.encoder = Encoder(trunk='resnet50', layer_to_freezing=opts['layer_to_freezing'], downsize_factor=opts['downsize_factor'], use_ppm=False, ppm_remain_dim=True, specify_output_channel=None, lateral_output_mode_layer=(1, 6))  # default, 2048
        print(self.encoder)

        if opts['use_domain_confusion']:
            self.auxiliary_classifier = ClassifierGRL(input_channel=2048, output_classes=4, architecture_type='type3')
            self.auxiliary_classifier_kp = ClassifierGRL(input_channel=1, output_classes=opts['N_way'], architecture_type='type3')
            print(self.auxiliary_classifier)
            print(self.auxiliary_classifier_kp)

        # used in feature modulator
        # self.context_mode = 'hard_fiber'
        # self.context_mode = 'soft_fiber_bilinear'
        self.context_mode = 'soft_fiber_gaussian'
        # self.context_mode = 'soft_fiber_gaussian2'

        # input feature map: 1024 x 23 x 23 (1/16), 2048 x 12 x 12 (1/32), 3072 x 8 x 8   (1/46), 4096 x 6 x 6   (1/64)
        self.descriptor_net = DescriptorNet(2048, 3, output_fiber=True, specify_fc_out_units=None, architecture_type='type3',   net_config=(512, 1024))
        print(self.descriptor_net)

        self.regression_type = 'direct_regression_gridMSE'

        if self.regression_type == 'direct_regression_gridMSE':
            self.grid_length = opts['grid_length']  # [4, 8, 12]
            modules = []
            covar_branch_module = []
            for grid_scale in self.grid_length:
                # a) independent variance or covariance for each kp
                modules += [GridBasedLocatorX2Covar(grid_length=grid_scale, reg_uncertainty=True, cls_uncertainty=False, ureg_type=1, covar_Q_size=(2,3), sample_times=opts['sample_times'], negative_noise=False, sigma_computing_mode=0, alpha=5, beta=0.2)]

                # b) covariance for all kps
                # covar_branch_module += [CovarNet(grid_length=grid_scale, covar_Q_size=(2, 2, 2, 3))]  # covar for 2 keypoints
                covar_branch_module += [CovarNet(grid_length=grid_scale, covar_Q_size=(3, 3, 2, 3))]  # covar for 3 keypoints


            self.regressor = nn.ModuleList(modules)  # main regressor
            self.covar_branch = nn.ModuleList(covar_branch_module)

        print(self.regressor)

        # if self.opts['use_pum'] == True:
        #     self.pum = PatchUncertaintyModule(input_channel=2048 * 2, conv_layers=2)  # 512, 2048, 3072, 4096


        # loading model based on the configurations said in self.opts
        self.load_model()

        self.loss_fun_mse = nn.MSELoss(reduction='sum')  # Used in keypoint regression
        # loss_fun_mse = nn.L1Loss(reduction='sum')
        self.loss_fun_nll = nn.NLLLoss(ignore_index=-1)  # Used in animal class classification
        if torch.cuda.is_available():
            self.loss_fun_mse = self.loss_fun_mse.cuda()
            self.loss_fun_nll = self.loss_fun_nll.cuda()


        self.optimizer_init(lr=0.0001, lr_auxiliary = 0.0001, weight_decay=0, optimization_algorithm='Adam')
        # self.optimizer_alex = optim.Adam(self.alexnet.parameters(), lr=0.0001)


        self.recall_stack=[0, 0]
        self.recall_best = 0  # used to record the best recall

        if self.opts['use_body_part_protos']:
            if self.opts['load_proto']:
                self.load_proto_memory()
            else:
                # compute the number of interpolated kps
                if self.opts['use_interpolated_kps'] == True:
                    auxiliary_paths = get_auxiliary_paths(self.opts['auxiliary_path_mode'], self.episode_generator_test.support_kp_categories)
                    N_paths = len(auxiliary_paths)
                    N_knots = len(self.opts['interpolation_knots'])
                    T = N_paths * N_knots
                else:
                    T = 0
                self.init_proto_memory(stat_episode_num=100, fiber_dim=2048, part_num=len(self.episode_generator_test.support_kp_categories), auxiliary_kp_num=T, method=self.opts['proto_compute_method'])
            if self.opts['memorize_fibers']:
                self.memorized_episode_cnt = 0
                torch.set_grad_enabled(False)  # disable grad computation when recording fibers
                print('Stop grad computation!')
                self.init_fiber_memory()

    def init_proto_memory(self, stat_episode_num=300, fiber_dim=2048, part_num=15, auxiliary_kp_num=0, method='ws'):
        self.memory={}
        self.memory['stat_episode_num'] = stat_episode_num
        self.memory['fiber_dim'] = fiber_dim
        self.memory['part_num'] = part_num
        self.memory['auxiliary_kp_num'] = auxiliary_kp_num
        self.memory['proto_compute_method'] = method  # 'm': mean; 'ws': weighted_sum
        self.memory['proto'] = torch.zeros(fiber_dim, part_num, requires_grad=False).cuda()  # C x N
        self.memory['proto_mask'] = torch.ones(part_num, requires_grad=False).cuda()  # N
        self.memory['aux_proto'] = torch.zeros(fiber_dim, auxiliary_kp_num, requires_grad=False).cuda()  # C x T
        self.memory['aux_proto_mask'] = torch.ones(auxiliary_kp_num, requires_grad=False).cuda()  # T

    def init_fiber_memory(self):
        episode_num = self.memory['stat_episode_num']
        fiber_dim = self.memory['fiber_dim']
        part_num = self.memory['part_num']
        auxiliary_kp_num = self.memory['auxiliary_kp_num']
        self.memory['fibers'] = torch.zeros(episode_num, fiber_dim, part_num, requires_grad=False).cuda()  # K x C x N, N parts
        self.memory['distance'] = torch.zeros(episode_num, part_num, requires_grad=False).cuda()  # K x N
        self.memory['mask'] = torch.zeros(episode_num, part_num, requires_grad=False).cuda()  # K x N
        if auxiliary_kp_num > 0:
            self.memory['aux_fibers'] = torch.zeros(episode_num, fiber_dim, auxiliary_kp_num, requires_grad=False).cuda()  # K x C x N, N parts
            self.memory['aux_distance'] = torch.zeros(episode_num, auxiliary_kp_num, requires_grad=False).cuda()  # K x N
            self.memory['aux_mask'] = torch.zeros(episode_num, auxiliary_kp_num, requires_grad=False).cuda()  # K x N

    def save_proto_memory(self, path=None):
        if path == None:
            # e.g., 'fiber_protos_and_mask_ws.pt', 'fiber_protos_and_mask_m.pt'
            path = 'animal_fiber_protos_and_mask_' + self.memory['proto_compute_method'] + '.pt'
        memory_dict = {}
        memory_dict['stat_episode_num'] = self.memory['stat_episode_num']
        memory_dict['proto_compute_method'] = self.memory['proto_compute_method']  # 'm': mean; 'ws': weighted_sum
        memory_dict['proto'] = self.memory['proto'].cpu()
        memory_dict['proto_mask'] = self.memory['proto_mask'].cpu()
        memory_dict['aux_proto'] = self.memory['aux_proto'].cpu()
        memory_dict['aux_proto_mask'] = self.memory['aux_proto_mask'].cpu()
        torch.save(memory_dict, path)

    def load_proto_memory(self, path=None):
        if path == None:
            # e.g., 'fiber_protos_and_mask_ws.pt', 'fiber_protos_and_mask_m.pt'
            path = 'animal_fiber_protos_and_mask_' + self.opts['proto_compute_method'] + '.pt'
        self.memory = {}
        memory_dict = torch.load(path)
        self.memory['proto'] = memory_dict['proto'].cuda()
        self.memory['proto_mask'] = memory_dict['proto_mask'].cuda()
        self.memory['aux_proto'] = memory_dict['aux_proto'].cuda()
        self.memory['aux_proto_mask'] = memory_dict['aux_proto_mask'].cuda()
        self.memory['stat_episode_num'] = memory_dict['stat_episode_num']
        fiber_dim, part_num = (self.memory['proto']).shape
        auxiliary_kp_num = len(self.memory['aux_proto_mask'])
        self.memory['fiber_dim'] = fiber_dim
        self.memory['part_num'] = part_num
        self.memory['auxiliary_kp_num'] = auxiliary_kp_num
        self.memory['proto_compute_method'] = self.opts['proto_compute_method']  # 'm': mean; 'ws': weighted_sum

    def train(self, *multiple_episode_generators: EpisodeGenerator):
        episode_i = 0
        sample_failure_cnt = 0   # count failedly sampled episodes
        using_multiple_episodes = False
        sample_failure_cnt2 = 0  # count those wrongly labeled images

        using_interpolated_kps = self.opts['use_interpolated_kps']
        interpolation_knots = self.opts['interpolation_knots']

        alpha = 0
        loss_adapt_total, loss_adapt_kp_total = 0, 0
        loss_symmetry_total = 0
        loss_interpolation_total = 0
        loss_total = 0

        # t_count = np.array([0]*7, dtype=np.float)
        if len(multiple_episode_generators) > 0:  # if multiple_episode_generators is not empty
            using_multiple_episodes = True
            num_multiple_episode = len(multiple_episode_generators)
            prob = np.array([len(multiple_episode_generators[i].samples) for i in range(num_multiple_episode)])
            prob = prob / np.sum(prob)
        while episode_i < self.opts['num_episodes']:
            if episode_i % 1600 == 0 and episode_i >= 0:
                eval_results = self.validate(self.episode_generator_test, eval_method=self.opts['eval_method'])
                recall = eval_results[0]  # parse eval results
                # recall, _, recall_aux, _ = self.validate2(self.episode_generator_test, eval_method=self.opts['eval_method'])  # testing for aux kps
                if self.episode_generator_test2 != None:
                    eval_results2 = self.validate(self.episode_generator_test2, eval_method=self.opts['eval_method'])
                    recall2 = eval_results2[0]  # parse eval results
                if self.writer != None:
                    self.writer.add_scalar('accuracy', recall[0], episode_i)  # recall is a list which is corresponding to different thresholds
                    if self.episode_generator_test2 != None:
                        self.writer.add_scalar('accuracy2', recall2[0], episode_i)

                # save model based on the configurations said in self.opts
                self.recall_stack[0] = self.recall_stack[1]
                self.recall_stack[1] = recall2[0]
                avg_recall = np.mean(self.recall_stack)
                if avg_recall > self.recall_best:
                    if episode_i >= 400:
                        self.save_model()
                    self.recall_best = avg_recall
                    print('BEST:', self.recall_best)

            if self.opts['use_body_part_protos'] and self.opts['memorize_fibers']:
                # disable grad computation when recording fibers
                # since each time the grad will be enabled after using self.validate, we have to put stop grad function here
                torch.set_grad_enabled(False)

            # roll-out an episode
            if using_multiple_episodes == False:  # training by using single episode generator, which is default case
                episode_generator = self.episode_generator
            else:
                random_episode_ind = np.random.randint(0, num_multiple_episode, 1)
                # random_episode_ind = np.random.choice(range(0, num_multiple_episode), size=1, p=prob)
                # print('index: ', random_episode_ind)
                episode_generator = multiple_episode_generators[random_episode_ind[0]]

            while (False == episode_generator.episode_next()):
                sample_failure_cnt += 1
                if sample_failure_cnt % 500 == 0:
                    print('sample failure times: {}'.format(sample_failure_cnt))
                continue
            # print(episode_generator.support_kp_categories)


            preprocess = mytransforms.Compose([
                # color transform
                # mytransforms.RandomApply(mytransforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), p=0.8),
                # mytransforms.RandomGrayscale(p=0.01),
                # geometry transform
                mytransforms.RandomApply(mytransforms.HFlip(swap=horizontal_swap_keypoints), p=0.5),  # 0.5
                mytransforms.RandomApply(mytransforms.RandomRotation(max_rotate_degree=15), p=0.25),  # 0.25
                mytransforms.RelativeResize((0.75, 1.25)),
                mytransforms.RandomCrop(crop_bbox=False),
                # mytransforms.RandomApply(mytransforms.RandomTranslation(), p=0.5),
                mytransforms.Resize(longer_length=self.opts['square_image_length']),  # 368
                mytransforms.CenterPad(target_size=self.opts['square_image_length']),
                mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=False)
            ])

            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # define a list containing the paths, each path is represented by kp index pair [index1, index2]
            # our paths are subject to support keypoints; then we will interpolated kps for each path. The paths is possible to be empty list []
            num_random_paths = self.opts['num_random_paths']  # only used when auxiliary_path_mode='random'
            # path_mode:  'exhaust', 'predefined', 'random'
            auxiliary_paths = get_auxiliary_paths(path_mode=self.opts['auxiliary_path_mode'], support_keypoint_categories=episode_generator.support_kp_categories, num_random_paths=num_random_paths)
            support_dataset = AnimalPoseDataset(episode_generator.supports,
                                                episode_generator.support_kp_categories,
                                                using_auxiliary_keypoints=using_interpolated_kps,
                                                interpolation_knots=interpolation_knots,
                                                interpolation_mode=self.opts['interpolation_mode'],
                                                auxiliary_path=auxiliary_paths,
                                                hdf5_images_path=self.opts['hdf5_images_path'],
                                                saliency_maps_root=self.opts['saliency_maps_root'],
                                                output_saliency_map=self.opts['use_pum'],
                                                preprocess=preprocess,
                                                input_transform=image_transform
                                                )

            query_dataset = AnimalPoseDataset(episode_generator.queries,
                                              episode_generator.support_kp_categories,
                                              using_auxiliary_keypoints=using_interpolated_kps,
                                              interpolation_knots=interpolation_knots,
                                              interpolation_mode=self.opts['interpolation_mode'],
                                              auxiliary_path=auxiliary_paths,
                                              hdf5_images_path=self.opts['hdf5_images_path'],
                                              saliency_maps_root=self.opts['saliency_maps_root'],
                                              output_saliency_map=self.opts['use_pum'],
                                              preprocess=preprocess,
                                              input_transform=image_transform
                                              )

            support_loader = DataLoader(support_dataset, batch_size=self.opts['K_shot'], shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.opts['M_query'], shuffle=False)

            support_loader_iter = iter(support_loader)
            query_loader_iter = iter(query_loader)
            (supports, support_labels, support_kp_mask, _, support_aux_kps, support_aux_kp_mask, support_saliency, _, _) = support_loader_iter.next()
            (queries, query_labels, query_kp_mask, _, query_aux_kps, query_aux_kp_mask, query_saliency, _, _) = query_loader_iter.next()

            # print_weights(support_kp_mask)
            # print_weights(query_kp_mask)
            # make_grid_images(supports, denormalize=True, save_path='grid_image_s.jpg')
            # make_grid_images(queries, denormalize=True, save_path='grid_image_q.jpg')
            # make_grid_images(support_saliency.cuda(), denormalize=False, save_path='./ss.jpg')
            # make_grid_images(query_saliency.cuda(), denormalize=False, save_path='./sq.jpg')
            # print(episode_generator.supports)
            ## 'exhaust', 'predefined'
            # save_episode_before_preprocess(episode_generator, episode_i, delete_old_files=False, draw_interpolated_kps=using_interpolated_kps, interpolation_knots=interpolation_knots, interpolation_mode=self.opts['interpolation_mode'], path_mode='predefined')
            # show_save_episode(supports, support_labels, support_kp_mask, queries, query_labels, query_kp_mask, episode_generator, episode_i,
            #                   support_aux_kps, support_aux_kp_mask, query_aux_kps, query_aux_kp_mask, is_show=False, is_save=True, delete_old_files=False)
            # show_save_episode(supports, support_labels, support_kp_mask, queries, query_labels, query_kp_mask, episode_generator, episode_i,
            #                   support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None, is_show=False, is_save=True, delete_old_files=False)
            # print_weights(episode_generator.support_kp_mask)
            # print_weights(episode_generator.query_kp_mask)

            # support_kp_mask = episode_generator.support_kp_mask  # B1 x N
            # query_kp_mask = episode_generator.query_kp_mask      # B2 x N

            # if torch.cuda.is_available():
            supports, queries = supports.cuda(), queries.cuda()  # B1 x C x H x W, B2 x C x H x W
            support_labels, query_labels = support_labels.float().cuda(), query_labels.float().cuda()  # B1 x N x 2, B2 x N x 2
            # support_labels.requires_grad = True

            support_kp_mask = support_kp_mask.cuda()  # B1 x N
            query_kp_mask = query_kp_mask.cuda()      # B2 x N
            if using_interpolated_kps:
                support_aux_kps = support_aux_kps.float().cuda()  # B1 x T x 2, T = (N_paths * N_knots), total number of auxiliary keypoints
                support_aux_kp_mask = support_aux_kp_mask.cuda()  # B1 x T
                query_aux_kps = query_aux_kps.float().cuda()      # B2 x T x 2
                query_aux_kp_mask = query_aux_kp_mask.cuda()      # B2 x T

            # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
            union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0  # N
            # compute the valid query keypoints, using broadcast
            valid_kp_mask = (query_kp_mask * union_support_kp_mask.reshape(1, -1))  # B2 x N
            num_valid_kps = torch.sum(valid_kp_mask)
            if using_interpolated_kps:
                num_valid_support_aux_kps = torch.sum(support_aux_kp_mask)  # only for support auxiliary kps
                union_support_aux_kp_mask = torch.sum(support_aux_kp_mask, dim=0) > 0  # T
                valid_aux_kp_mask = query_aux_kp_mask * union_support_aux_kp_mask.reshape(1, -1)  # B x T
                num_valid_aux_kps = torch.sum(valid_aux_kp_mask)
                # print(num_valid_kps+num_valid_aux_kps)

            num_valid_kps_for_samples = torch.sum(valid_kp_mask, dim=1)  # B2
            gp_valid_samples_mask = num_valid_kps_for_samples >= episode_generator.least_query_kps_num  # B2
            gp_num_valid_kps = torch.sum(num_valid_kps_for_samples * gp_valid_samples_mask)
            gp_valid_kp_mask = valid_kp_mask * gp_valid_samples_mask.reshape(-1, 1)  # B2 x N

            # check #1    there may exist some wrongly labeled images where the keypoints are outside the boundary
            if torch.any(support_labels > 1) or torch.any(support_labels < -1) or torch.any(query_labels > 1) or torch.any(query_labels < -1):
                sample_failure_cnt2 += 1
                if (sample_failure_cnt2 % 50 == 0):
                    print('count for wrongly labelled images: {}'.format(sample_failure_cnt))
                continue             # skip current episode directly
            # check #2
            if num_valid_kps ==  0:  # flip transform may lead zero intersecting keypoints between support and query, namely zero valid kps
                continue             # skip current episode directly




            if self.encoder_type == 0:  # Encoder, EncoderMultiScaleType1
                # feature and semantic distinctiveness, note that x2 = support_saliency.cuda() or query_saliency.cuda()
                support_features, support_lateral_out = self.encoder(x=supports, x2=None, enable_lateral_output=True)  # B1 x C x H x W, B1 x 1 x H x W
                query_features, query_lateral_out = self.encoder(x=queries, x2=None, enable_lateral_output=True)  # B2 x C x H x W, B2 x 1 x H x W
                if self.opts['use_pum']:
                    # computing_inv_w = True
                    w_computing_mode = 2
                    p_support_lateral_out = self.numerical_transformation(support_lateral_out, w_computing_mode=w_computing_mode)  # B1 x 1 x H' x W', transform into positive number
                    p_query_lateral_out = self.numerical_transformation(query_lateral_out, w_computing_mode=w_computing_mode)  # B2 x 1 x H' x W', transform into positive number

                # B1 x C x N
                support_repres, conv_query_features = extract_representations(support_features, support_labels, support_kp_mask, context_mode=self.context_mode,
                                sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'], together_trans_features=None)
                avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
                attentive_features, attention_maps_l2, attention_maps_l1, similarity_map = feature_modulator3(avg_support_repres, query_features, \
                                       fused_attention=self.opts['use_fused_attention'], output_attention_maps=False, compute_similarity=False)

                # loss_similarity = torch.tensor(0)  # self.compute_similarity_loss(similarity_map, query_labels, valid_kp_mask)


                # show_save_attention_maps(attention_maps_l2, queries, support_kp_mask, query_kp_mask, episode_generator.support_kp_categories,
                #                          episode_num=episode_i, is_show=False, is_save=True, delete_old=False, T=query_scale_trans)
                # show_save_attention_maps(attention_maps_l1, queries, support_kp_mask, query_kp_mask, episode_generator.support_kp_categories,
                #                          episode_num=episode_i, is_show=False, is_save=True, save_image_root='./attention_maps2', delete_old=False, T=query_scale_trans)


                if using_interpolated_kps and num_valid_support_aux_kps > 0:
                    # B1 x C x M
                    support_repres_aux, _ = extract_representations(support_features, support_aux_kps, support_aux_kp_mask, context_mode=self.context_mode,
                                       sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'])
                    avg_support_repres_aux = average_representations2(support_repres_aux, support_aux_kp_mask)
                    attentive_features_aux, _, _, _ = feature_modulator3(avg_support_repres_aux, query_features, \
                                   fused_attention=self.opts['use_fused_attention'], output_attention_maps=False, compute_similarity=False)

                if self.opts['use_pum']:  # extracting semantic distinctiveness (SD) per keypoint (i.e., modeling semantic uncertainty)
                    # B1 x C x T
                    # support_saliency_repres_aux, _ = extract_representations(support_saliency_features, support_aux_kps, support_aux_kp_mask, context_mode=self.context_mode,
                    #                    sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'])
                    # support_patches_features = torch.cat([support_repres_aux, support_saliency_repres_aux], dim=1)  # B1 x (2C) x T
                    # patches_rho_aux_origin = self.pum(support_patches_features)  # B1 x T, T is the number of interpolated keypoints
                    # # patches_rho may need to do average to get T values, here we asssume the shot number is 1
                    # # patches_rho_aux = patches_rho_aux_origin.mean(dim=0)  # T
                    # B1, T = patches_rho_aux_origin.shape
                    # patches_rho_aux = average_representations(patches_rho_aux_origin.view(B1, 1, T), support_aux_kp_mask).squeeze()

                    # regarding main training kps
                    # inv_w_patches, w_patches = self.get_distinctiveness_for_parts(p_support_lateral_out, p_query_lateral_out, support_labels, support_kp_mask, query_labels)
                    inv_w_patches = None

                    # p = float(episode_i) / self.opts['num_episodes']
                    # alpha = 2. / (1. + np.exp(-10 * p)) - 1  # ranges from 0 to 1
                    # # support_labels_for_pum = ReverseLayerF.apply(support_labels, alpha)
                    # # query_labels_for_pum = ReverseLayerF.apply(query_labels, alpha)
                    # support_labels_for_pum = support_labels.detach()
                    # query_labels_for_pum = query_labels.detach()
                    # inv_w_patches, w_patches = self.get_distinctiveness_for_parts(p_support_lateral_out, p_query_lateral_out, support_labels_for_pum, support_kp_mask, query_labels_for_pum)


                    # regarding auxiliary kps
                    if using_interpolated_kps and num_valid_support_aux_kps > 0:
                        inv_w_patches_aux, w_patches_aux = self.get_distinctiveness_for_parts(p_support_lateral_out, p_query_lateral_out, support_aux_kps, support_aux_kp_mask, query_aux_kps)


                        # # support_aux_kps_for_pum = ReverseLayerF.apply(support_aux_kps, alpha)
                        # # query_aux_kps_for_pum = ReverseLayerF.apply(query_aux_kps, alpha)
                        # support_aux_kps_for_pum = support_aux_kps.detach()
                        # query_aux_kps_for_pum = query_aux_kps.detach()
                        # inv_w_patches_aux, w_patches_aux = self.get_distinctiveness_for_parts(p_support_lateral_out, p_query_lateral_out, support_aux_kps_for_pum, support_aux_kp_mask, query_aux_kps_for_pum)

                    else:
                        inv_w_patches_aux, w_patches_aux = None, None
                else:
                    # patches_rho_aux = None
                    inv_w_patches_aux, w_patches_aux = None, None
                    inv_w_patches, w_patches = None, None



            elif self.encoder_type == 1:  # can only converge when using feature_modulator2()
                attentive_features = self.encoder(supports, queries, support_labels, support_kp_mask)

            keypoint_descriptors = self.descriptor_net(attentive_features)  # B2 x N x D  or B2 x N x c x h x w
            if using_interpolated_kps and num_valid_support_aux_kps > 0:
                keypoint_descriptors_aux = self.descriptor_net(attentive_features_aux)  # B2 x T x D or B2 x T x c x h x w

            if self.regression_type == 'direct_regression_gridMSE':
                B2 = self.opts['M_query']
                N = self.opts['N_way']
                w_dev = 1.0
                if w_dev ==0 and self.opts['loss_weight'][2] == 0:  # because cls uses w (semantic uncertainty) to weight loss
                    inv_w_patches = inv_w_patches_aux = None

                # don't use interpolated kps or don't have interpolated kps
                if using_interpolated_kps == False or (using_interpolated_kps == True and num_valid_support_aux_kps == 0):
                    loss_grid_class, loss_deviation, mean_predictions = self.multiscale_regression(keypoint_descriptors, query_labels, valid_kp_mask, self.grid_length, weight=None, inv_w_patches=inv_w_patches, w_patches=None)
                    loss = loss_grid_class + w_dev * loss_deviation

                    loss_interpolation = torch.tensor(0.).cuda()
                    loss_aux_grid, loss_aux_deviation = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
                    loss_covar_multiple_kps, loss_covar_multiple_kps_total = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
                    loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()
                    loss_offset = torch.tensor(0.).cuda()

                else:  # using interpolated kps
                    T = valid_aux_kp_mask.shape[1]  # T = (N_knots * N_paths)
                    N_curves = (int)(T / len(interpolation_knots))
                    # aux_kps_weights1 = torch.ones(2*3).cuda()
                    # aux_kps_weights2 = torch.tensor([0.75, 0.5, 0.75]).repeat(N_curves-2, 1).reshape(-1).cuda()
                    # aux_kps_weights2 = torch.tensor([0.75, 0.25, 0.75]).repeat(N_curves-2, 1).reshape(-1).cuda()
                    # aux_kps_weights  =  torch.cat([aux_kps_weights1, aux_kps_weights2], dim=0)
                    # aux_kps_weights = torch.tensor([0.75, 0.5, 0.75]).repeat(N_curves, 1).reshape(-1).cuda()
                    # aux_kps_weights = torch.tensor([0.75, 0.25, 0.75]).repeat(N_curves, 1).reshape(-1).cuda()
                    aux_kps_weights = None

                    loss_grid_class, loss_deviation, loss_aux_grid, loss_aux_deviation, loss_covar_multiple_kps, loss_covar_multiple_kps_aux, mean_predictions, mean_predictions_aux = \
                        self.multiscale_regression2(keypoint_descriptors, query_labels, valid_kp_mask, keypoint_descriptors_aux, query_aux_kps, valid_aux_kp_mask, self.grid_length, \
                                                    episode_generator.support_kp_categories, query_dataset.auxiliary_paths, weight=None, inv_w_patches_aux=inv_w_patches_aux, w_patches_aux=None, inv_w_patches=inv_w_patches, w_patches=None)

                    # ---------------main keypoints-------------------
                    loss = loss_grid_class + w_dev * loss_deviation
                    # ---------------auxiliary keypoints-------------------
                    loss_interpolation = loss_aux_grid + w_dev * loss_aux_deviation
                    loss_interpolation_total += loss_interpolation

                    loss_covar_multiple_kps_total = loss_covar_multiple_kps + loss_covar_multiple_kps_aux

            # square_image_length = self.opts['square_image_length']
            # show_save_predictions(queries, query_labels.cpu().detach(), valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=None,
            #                       version='original', is_show=False, is_save=True, folder_name='query_gt', save_file_prefix='eps{}'.format(episode_i))
            # show_save_predictions(queries, mean_predictions.cpu().detach(), valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=None, confidence_scale=3,
            #                       version='original', is_show=False, is_save=True, folder_name='query_predict', save_file_prefix='eps{}'.format(episode_i))

            if self.opts['use_body_part_protos']:
                if self.opts['memorize_fibers']:
                    if episode_i <= self.memory['stat_episode_num'] - 1:  # record fibers
                        # rectify the current body part order to the standard order in case "order_fixed = False", namely the dynamic support kp categories
                        order_index = [episode_generator.support_kp_categories.index(kp_type) for kp_type in KEYPOINT_TYPES]
                        self.memory['fibers'][episode_i] = (avg_support_repres[:, order_index]).detach().clone()  # C x N
                        kp_weights = valid_kp_mask.sum(dim=0)     # N
                        d_sum = torch.sum((mean_predictions.detach() - query_labels)**2, dim=2)  # B x N
                        d     = torch.sum(d_sum * valid_kp_mask, dim=0) / (kp_weights+1e-6)  # N
                        self.memory['distance'][episode_i] = d[order_index]
                        self.memory['mask'][episode_i] = (kp_weights > 0)[order_index]  # N

                        # here we suppose the aux kp order is fixed (should set the 'order_fixed = True' in main.py)
                        if self.memory['auxiliary_kp_num'] > 0 and num_valid_support_aux_kps > 0:  # in case there is no aux kps
                            self.memory['aux_fibers'][episode_i] = avg_support_repres_aux.detach().clone()  # C x T
                            kp_weights_aux = valid_aux_kp_mask.sum(dim=0)  # T
                            d_sum_aux = torch.sum((mean_predictions_aux.detach() - query_aux_kps) ** 2, dim=2)  # B x T
                            d_aux = torch.sum(d_sum_aux * valid_aux_kp_mask, dim=0) / (kp_weights_aux + 1e-6)  # T
                            self.memory['aux_distance'][episode_i] = d_aux
                            self.memory['aux_mask'][episode_i] = (kp_weights_aux > 0)  # T

                    if episode_i == self.memory['stat_episode_num'] - 1:  # build body part prototypes which is subject the standard order
                        d_total = self.memory['distance']  # K x N, l2 distance
                        d_total = torch.sqrt(d_total)      # K x N, square root
                        m_total = self.memory['mask']      # K x N
                        masked_exp_neg_d = torch.exp(-d_total) * m_total
                        p_total = masked_exp_neg_d / torch.sum(masked_exp_neg_d, dim=0).view(1, -1)  # K x N
                        self.memory['proto_mask'] = m_total.sum(dim=0) > 0  # N

                        if self.memory['auxiliary_kp_num'] > 0 and num_valid_support_aux_kps > 0:  # in case there is no aux kps
                            d_total_aux = self.memory['aux_distance']  # K x T, l2 distance
                            d_total_aux = torch.sqrt(d_total_aux)  # K x T, square root
                            m_total_aux = self.memory['aux_mask']  # K x T
                            masked_exp_neg_d_aux = torch.exp(-d_total_aux) * m_total_aux
                            p_total_aux = masked_exp_neg_d_aux / torch.sum(masked_exp_neg_d_aux, dim=0).view(1, -1)  # K x T
                            self.memory['aux_proto_mask'] = m_total_aux.sum(dim=0) > 0  # N

                        if self.memory['proto_compute_method'] == 'ws':  # weighted sum
                            # ---
                            # Method 1, use prediction's deviation to GT to serve as weight
                            universal_body_part_protos = self.memory['fibers'] * p_total.view(self.memory['stat_episode_num'], 1, self.memory['part_num'])  # K x C x N
                            self.memory['proto'] = torch.sum(universal_body_part_protos, dim=0)  # C x N

                            if self.memory['auxiliary_kp_num'] > 0 and num_valid_support_aux_kps > 0:  # in case there is no aux kps
                                universal_protos_aux = self.memory['aux_fibers'] * p_total_aux.view(self.memory['stat_episode_num'], 1, self.memory['auxiliary_kp_num'])  # K x C x T
                                self.memory['aux_proto'] = torch.sum(universal_protos_aux, dim=0)  # C x T
                        else:  # mean
                            # ---
                            # Method 2, simple mean
                            universal_body_part_protos = self.memory['fibers'] * m_total.view(self.memory['stat_episode_num'], 1, self.memory['part_num'])  # K x C x N
                            self.memory['proto'] = torch.sum(universal_body_part_protos, dim=0) / (m_total.sum(dim=0) + 1e-6).view(1, -1)  # C x N

                            if self.memory['auxiliary_kp_num'] > 0 and num_valid_support_aux_kps > 0:  # in case there is no aux kps
                                universal_protos_aux = self.memory['aux_fibers'] * m_total_aux.view(self.memory['stat_episode_num'], 1, self.memory['auxiliary_kp_num'])  # K x C x T
                                self.memory['aux_proto'] = torch.sum(universal_protos_aux, dim=0) / (m_total_aux.sum(dim=0) + 1e-6).view(1, -1)  # C x T
                            #---
                        self.save_proto_memory()
                        torch.set_grad_enabled(True)  # enable grad computation when finishing recording fibers
                        print('Open grad computation!')
                        print('Initial universal body part prototypes built!')
                        exit(0)     # exit when body part protos are built
                    print(episode_i)
                    episode_i += 1  # skip loss functions and backwards, jump to next episode
                    continue

            loss_total += loss
            if episode_i % 1 == (1 - 1):
                loss_interpolation_total /= 1.0
                loss_total               /= 1.0

                final_combined_loss = self.opts['loss_weight'][0]*loss_total + self.opts['loss_weight'][1]*loss_interpolation_total + self.opts['loss_weight'][2]*loss_covar_multiple_kps_total # + loss_similarity # + 0.1 * loss_symmetry_total + 0.01*loss_adapt_total + 0.01*loss_adapt_kp_total
                #=================
                # sometimes will happen due to sdm (semantic distinctiveness) when beta_semantic = 0
                if torch.isnan(final_combined_loss):
                    loss_interpolation_total = 0
                    loss_total = 0
                    print('loss to be nan')
                    continue
                #==================

                self.optimizer_step(final_combined_loss)

                self.lr_scheduler_step(episode_i)

                loss_interpolation_total = 0
                loss_total = 0

            if episode_i % 8 == (8 - 1):
                # print('time: {}'.format(datetime.datetime.now()))
                if self.regression_type == 'direct_regression_gridMSE':
                    print('episode: {}, loss_kp: {:.5f} (G: {:.5f}/D: {:.5f}), loss_aux: {:.5f} (G: {:.5f}/D: {:.5f}), mcovar: {:.5f} (M: {:.5f}/A: {:.5f}), time: {}'.format(episode_i, loss.item(), loss_grid_class.item(),
                           loss_deviation.item(), loss_interpolation.item(), loss_aux_grid.item(), loss_aux_deviation.item(), loss_covar_multiple_kps_total.item(), loss_covar_multiple_kps.item(), loss_covar_multiple_kps_aux.item(),\
                           datetime.datetime.now()))

                    if self.writer != None:
                        self.writer.add_scalar('loss', loss.cpu().detach().numpy(), episode_i)

            # increment in episode_i
            episode_i += 1

    def multiscale_regression(self, keypoint_descriptors, query_kp_label, valid_kp_mask, grid_length_list, weight=None, inv_w_patches=None, w_patches=None):
        B2 = query_kp_label.shape[0]  # B2 x N x 2, N keypoints for each image
        N = query_kp_label.shape[1]
        num_valid_kps = torch.sum(valid_kp_mask)

        loss_grid_class = 0
        loss_deviation = 0
        mean_predictions = 0
        scale_num = len(grid_length_list)
        for scale_i, grid_length in enumerate(grid_length_list):
            # compute grid groundtruth and deviation
            gridxy = (query_kp_label /2 + 0.5) * grid_length  # coordinate -1~1 --> 0~self.grid_length, B2 x N x 2
            gridxy_quantized = gridxy.long().clamp(0, grid_length - 1)  # B2 x N x 2
            # Method 1, deviation range: -1~1
            label_deviations = (gridxy - (gridxy_quantized + 0.5)) * 2  # we hope the deviation ranges -1~1, B2 x N x 2
            # Method 2, deviation range: 0~1
            # label_deviations = (gridxy - gridxy_quantized)  # we hope the deviation ranges 0~1, B2 x N x 2
            label_grids = gridxy_quantized[:, :, 1] * grid_length + gridxy_quantized[:, :, 0]  # 0 ~ grid_length * grid_length - 1, B2 x N

            one_hot_grid_label = torch.zeros(B2, N, grid_length**2).cuda()  # B2 x N x (grid_length * grid_length)
            one_hot_grid_label = one_hot_grid_label.scatter(dim=2, index=torch.unsqueeze(label_grids, dim=2), value=1)  # B2 x N x (grid_length * grid_length)
            # 1) global deviation
            # predict_grids, predict_deviations, rho, _ = self.regressor[scale_i](keypoint_descriptors, training_phase=True, one_hot_grid_label=one_hot_grid_label)  # B2 x N x (grid_length ** 2), B2 x N x 2, B2 x N x 2 (or B2 x N x 2d)
            # 2) local deviation
            predict_grids, predict_deviations, rho, _ = self.regressor[scale_i](keypoint_descriptors, training_phase=True, one_hot_grid_label=label_grids)  # B2 x N x (grid_length ** 2), B2 x N x 2, B2 x N x 2 (or B2 x N x 2d)

            
            # compute grid classification loss and deviation loss
            # predict_grids2 = predict_grids * valid_kp_mask.view(B2, N, 1)
            # predict_grids2 = predict_grids2.view(B2 * N, -1)
            # label_grids2 = (label_grids * valid_kp_mask).long()
            # label_grids2 = label_grids2.view(-1)
            predict_grids2 = predict_grids.view(B2 * N, -1)   # (B2 * N) * (grid_length * grid_length)
            label_grids2 = label_grids
            for i in range(B2):
                for j in range(N):
                    if valid_kp_mask[i, j] < 1:
                        label_grids2[i, j] = -1  # set ignore index for nllloss
            label_grids2 = label_grids2.view(-1)  # (B2 * N)

            # loss_grid_class += self.loss_fun_nll(predict_grids2, label_grids2)
            # # type1, direct regressing deviation
            # predict_deviations2 = predict_deviations * valid_kp_mask.view(B2, N, 1)
            # label_deviations2 = label_deviations * valid_kp_mask.view(B2, N, 1)
            # loss_deviation += self.loss_fun_mse(predict_deviations2, label_deviations2)

            if weight is None:

                # A) main training kps, grid classification
                if inv_w_patches is None:
                    loss_grid_class += self.loss_fun_nll(predict_grids2, label_grids2)
                else:
                    # loss_grid_class += self.loss_fun_nll(predict_grids2, label_grids2)

                    square_root_for_w = True

                    # # using patches uncertainty into consideration
                    if square_root_for_w == False:
                        if len(inv_w_patches.shape) == 1:  # only use support patches to compute inv_w, size is N
                            instance_weight = inv_w_patches.repeat(B2, 1).reshape(-1)  # (B2*N)
                        elif len(inv_w_patches.shape) == 2:  # use both support & query patches, size is B2 x N
                            instance_weight = inv_w_patches.reshape(-1)  # (B2*N)
                    else:
                        if len(inv_w_patches.shape) == 1:  # only use support patches to compute inv_w, size is N
                            instance_weight = torch.sqrt(inv_w_patches).repeat(B2, 1).reshape(-1)
                        elif len(inv_w_patches.shape) == 2:  # use both support & query patches, size is B2 x N
                            instance_weight = torch.sqrt(inv_w_patches).reshape(-1)

                    loss_grid_class += instance_weighted_nllloss(predict_grids2, label_grids2, instance_weight=instance_weight, ignore_index=-1)  # weight: (B2 * N)
                    # compute penatly for weights
                    # penalty = torch.sum(log_w_patches.repeat(B2, 1) * valid_kp_mask_aux)  # use log(w) as penalty
                    # if num_valid_kps_aux > 0:
                    #     penalty /= num_valid_kps_aux
                    # # print(penalty)
                    # loss_grid_class_aux += penalty
                    # ---------------------------------------
            else:
                instance_weight = weight.repeat(B2, 1).reshape(-1)  # construct instance_weights which has (B*N) elements
                loss_grid_class += instance_weighted_nllloss(predict_grids2, label_grids2, instance_weight=instance_weight, ignore_index=-1)  # weight: N

            # ------------------
            if weight is None:
                if inv_w_patches is None:
                    # type1, direct regressing deviation
                    # independent variance
                    # loss_deviation += masked_nll_gaussian(predict_deviations, label_deviations, rho, valid_kp_mask.view(B2, N, 1))
                    # loss_deviation += masked_nll_laplacian(predict_deviations, label_deviations, rho, valid_kp_mask.view(B2, N, 1))

                    # covariance for each keypoint
                    loss_deviation += masked_nll_gaussian_covar(predict_deviations, label_deviations, rho, valid_kp_mask)

                    # covariance for all keypoints
                    # loss_deviation += masked_nll_gaussian_covar2(predict_deviations, label_deviations, rho, valid_kp_mask, computing_mode=1)

                    # no variance
                    # loss_deviation += masked_mse_loss(predict_deviations, label_deviations, valid_kp_mask.view(B2, N, 1))
                    # loss_deviation += masked_l1_loss(predict_deviations, label_deviations, valid_kp_mask.view(B2, N, 1))

                else:
                    # loss_deviation += masked_l1_loss(predict_deviations, label_deviations, valid_kp_mask.view(B2, N, 1))
                    # loss_deviation += masked_nll_laplacian(predict_deviations, label_deviations, rho, valid_kp_mask.view(B2, N, 1))

                    # covariance for each keypoint
                    loss_fun_mode = 0
                    penalty_mode = 0  # 0 or 2 are better, namely log(det(W^-1)) or 1/ W^-1 - 1
                    beta = 1.0
                    beta_loc_uc = 1.0
                    loss_deviation += masked_nll_gaussian_covar(predict_deviations, label_deviations, rho, valid_kp_mask, patches_rho=inv_w_patches, loss_fun_mode=loss_fun_mode, penalty_mode=penalty_mode, beta=beta, beta_loc_uc=beta_loc_uc)

                    # using patches uncertainty into consideration
                    # loss_deviation += masked_nll_laplacian(predict_deviations, label_deviations, patches_rho.view(1, N, 1), valid_kp_mask.view(B2, N, 1), beta=0.5, computing_mode=0, offset1=6, offset2=0.5)
                    # loss_deviation += masked_nll_gaussian(predict_deviations, label_deviations, patches_rho.view(1, N, 1), valid_kp_mask.view(B2, N, 1), beta=0.5, computing_mode=0, offset1=6, offset2=0.5)
                    # loss_deviation += masked_nll_laplacian2(predict_deviations, label_deviations, rho, patches_rho.view(1, N, 1), valid_kp_mask.view(B2, N, 1), beta=0.5, gamma=0.98)
                    # loss_deviation += masked_nll_gaussian2(predict_deviations, label_deviations, rho, patches_rho.view(1, N, 1), valid_kp_mask.view(B2, N, 1), beta=0.5, gamma=0.95)
            else:
                # weight: N
                combine_mask_weight = (valid_kp_mask * weight.reshape(1, -1)).view(B2, N, 1)
                # loss_deviation += masked_nll_gaussian(predict_deviations, label_deviations, rho, combine_mask_weight)
                # loss_deviation += masked_nll_laplacian(predict_deviations, label_deviations, rho, combine_mask_weight)
                # loss_deviation += masked_mse_loss(predict_deviations, label_deviations, combine_mask_weight)
                loss_deviation += masked_l1_loss(predict_deviations, label_deviations, combine_mask_weight)

            # ------------------
            # type2, transform to original location
            # mean_predictions += (((gridxy_quantized + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2

            # compute predicted keypoint locations using grids and deviations
            out_predict_grids = torch.max(predict_grids, dim=2)[1]  # B2 x N, 0 ~ grid_length * grid_length - 1
            out_predict_gridxy = torch.FloatTensor(B2, N, 2).cuda()
            out_predict_gridxy[:, :, 0] = out_predict_grids % grid_length  # grid x
            out_predict_gridxy[:, :, 1] = out_predict_grids // grid_length # grid y

            # Method 1, deviation range: -1~1
            predictions = (((out_predict_gridxy + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
            # Method 2, deviation range: 0~1
            # predictions = ((predict_gridxy + predict_deviations) / grid_length - 0.5) * 2
            mean_predictions += predictions
            # ------------------

        # ------------------
        # type2, transform to original location
        mean_predictions /= scale_num
        # query_labels2 = query_labels * valid_kp_mask.view(B2, N, 1)
        # mean_predictions2 = mean_predictions * valid_kp_mask.view(B2, N, 1)
        # loss_deviation  = self.loss_fun_mse(mean_predictions2, query_labels2)
        # loss_deviation = loss_deviation / num_valid_kps * 15
        # ------------------

        loss_grid_class /= scale_num  # multi-scale
        # ------------------
        # type1, direct regressing deviation
        loss_deviation  /= scale_num  # multi-scale

        # if weight is None:
        #     num_valid_kps = torch.sum(valid_kp_mask).item()
        # else:
        #     num_valid_kps = torch.sum(valid_kp_mask * weight.reshape(1, -1)).item()
        # if num_valid_kps == 0:  # for symmetry case, it may not have the valid symmetric keypoints
        #     pass  # no need to divide symmetry_num_valid_kps since all are zero
        # else:
        #     loss_deviation = loss_deviation / num_valid_kps
        # ------------------

        # loss = loss_grid_class  + loss_deviation
        # print('loss_d: ', loss_deviation.cpu().detach().numpy())
        return loss_grid_class, loss_deviation, mean_predictions

    def construct_patches_grids(self, kp_labels, kp_mask, patch_size=3, interval_control=(2 / 12)):
        '''
        :param kp_labels: B x T x 2
        :param kp_mask:   B x T
        :param patch_size:
        :param interval_control:
        :return patch_grids, B x (T*patch_size*patch_size) x 2, e.g., B x (T*9) x 2
        :return patch_grids_mask, B x (T*patch_size*patch_size), e.g., B x (T*9)
        '''
        B, T = kp_labels.shape[:2]
        per_patch_grids_num = patch_size ** 2
        patch_grids = torch.zeros(B, T * per_patch_grids_num, 2, requires_grad=False).cuda()
        # patch_grids_mask = torch.zeros(B, T * per_patch_grids_num, requires_grad=False).cuda()
        half_patch_size = patch_size // 2
        for i in range(B):
            for j in range(T):
                if kp_mask[i, j] == 0:
                    continue
                cnt = 0  # 0~per_patch_grids_num-1
                for r in range(-half_patch_size, half_patch_size + 1, 1):
                    for c in range(-half_patch_size, half_patch_size + 1, 1):
                        y = kp_labels[i, j, 1] + r * interval_control
                        x = kp_labels[i, j, 0] + c * interval_control
                        patch_grids[i, j * per_patch_grids_num + cnt, 0] = x
                        patch_grids[i, j * per_patch_grids_num + cnt, 1] = y

                        cnt += 1
        return patch_grids

    def construct_patches_grids2(self, kp_labels, kp_mask, patch_size=3, interval_control=(2/12), label_max=1, label_min=-1):
        '''
        :param kp_labels: B x T x 2
        :param kp_mask:   B x T
        :param patch_size:
        :param interval_control:
        :return patch_grids, B x (T*patch_size*patch_size) x 2, e.g., B x (T*9) x 2
        :return patch_grids_mask, B x (T*patch_size*patch_size), e.g., B x (T*9)
        '''
        B, T = kp_labels.shape[:2]
        per_patch_grids_num = patch_size ** 2
        patch_grids = torch.zeros(B, T*per_patch_grids_num, 2, requires_grad=False).cuda()
        patch_grids_mask = torch.zeros(B, T*per_patch_grids_num, requires_grad=False).cuda()
        half_patch_size = patch_size // 2
        for i in range(B):
            for j in range(T):
                if kp_mask[i, j] == 0:
                    continue
                cnt = 0  # 0~per_patch_grids_num-1
                for r in range(-half_patch_size, half_patch_size+1, 1):
                    for c in range(-half_patch_size, half_patch_size+1, 1):
                        y = kp_labels[i, j, 1] + r * interval_control
                        x = kp_labels[i, j, 0] + c * interval_control
                        patch_grids[i, j * per_patch_grids_num + cnt, 0] = x
                        patch_grids[i, j * per_patch_grids_num + cnt, 1] = y

                        # build mask for generated grids
                        if x >= label_min and x <= label_max and y >= label_min and y <= label_max:
                            patch_grids_mask[i, j * per_patch_grids_num + cnt] = 1

                        cnt += 1
        return patch_grids, patch_grids_mask

    def compute_average_for_patch(self, map, points_set, keypoint_num, patch_size_square=9):
        '''
        :param points_set: B x (keypoint_num * patch_size_square) x 2
        :param map: B x 1 x H x W
        :return:
        '''
        values_set, _ = extract_representations(map, points_set, None, context_mode='soft_fiber_bilinear')  # B x 1 x (keypoint_num * patch_size_square)
        average = torch.mean(values_set.view(-1, keypoint_num, patch_size_square), dim=2)  # B x keypoint_num
        return average

    def find_valid_pair_kps(self, mask):
        '''
        Given kp mask, find valid pair of indices for valid keypoints
        :param mask: B x N
        :return: valid_inds, k x 3, each row is something like (0, 1, 3), which means kp indices (1, 3) in image 0 is a valid pair
        '''
        valid_inds_list = []
        B, N = mask.shape
        for i in range(B):
            for j in range(0, N-1, 1):
                if mask[i, j] == 0:
                    continue
                for k in range(j+1, N, 1):
                    if mask[i, k] == 0:
                        continue
                    valid_inds_list.append([i, j, k])
        valid_inds = torch.Tensor(valid_inds_list).long().cuda()

        return valid_inds  # k x 3

    def find_valid_pair_neighbor_kps(self, main_kp_mask, auxiliary_kp_mask, support_kp_categories, auxiliary_path):
        '''
        Given kp mask, find valid pairs of neighboring kps
        :param main_kp_mask: B x N
        :param auxiliary_kp_mask: B x T, T = (N_paths * N_knots)
        :param support_kp_categories:
        :param auxiliary_path: N_paths x 2, each row is a pair of body parts, original kp ids in KEYPOINT_TYPE
        :return: valid inds, k x 3
        '''
        B, N = main_kp_mask.shape
        _, T = auxiliary_kp_mask.shape
        interpolation_knots = self.opts['interpolation_knots']
        N_knots = len(interpolation_knots)  # interpolated knots
        N_paths = int(T / N_knots)  # the number of body part pairs served as path for interpolation
        # num_valid_kps_aux = torch.sum(auxiliary_kp_mask).long().item()
        valid_inds_list = []  # Each path has N_knots+1 valid pairs when this path is valid

        auxiliary_path_in_support = []
        for path_ind in range(N_paths):
            start_kp_id, end_kp_id = auxiliary_path[path_ind]
            start_kp_type, end_kp_type = KEYPOINT_TYPES[start_kp_id], KEYPOINT_TYPES[end_kp_id]
            start_kp_id_in_support, end_kp_id_in_support = support_kp_categories.index(start_kp_type), support_kp_categories.index(end_kp_type)
            auxiliary_path_in_support.append([start_kp_id_in_support, end_kp_id_in_support])

        for i in range(B):
            for j in range(T):
                if auxiliary_kp_mask[i, j] == 0:
                    continue
                knot_ind = j % N_knots
                if knot_ind == 0:  # first knot
                    path_ind = j // N_knots  # 0 ~ N_paths-1
                    start_kp_id_in_support, end_kp_id_in_support = auxiliary_path_in_support[path_ind]
                    valid_inds_list.append([i, start_kp_id_in_support, N+j])
                    if N_knots >= 2:
                        valid_inds_list.append([i, N + j, N + j + 1])
                    else:  # N_knots == 1, for this case the first knot is also the last knot
                        valid_inds_list.append([i, N + j, end_kp_id_in_support])
                elif knot_ind == N_knots-1:  # last knot
                    path_ind = j // N_knots  # 0 ~ N_paths-1
                    start_kp_id_in_support, end_kp_id_in_support = auxiliary_path_in_support[path_ind]
                    valid_inds_list.append([i, N+j, end_kp_id_in_support])
                else:  # middle knots
                    valid_inds_list.append([i, N+j, N+j+1])

        valid_inds = torch.Tensor(valid_inds_list).long().cuda()
        return valid_inds  # k x 3

    def find_valid_triplet_kps(self, main_kp_mask, auxiliary_kp_mask, support_kp_categories, auxiliary_path):
        '''
        Given kp mask, find valid triplets of indices
        :param main_kp_mask: B x N
        :param auxiliary_kp_mask: B x T, T = (N_paths * N_knots)
        :param auxiliary_path: N_paths x 2, each row is a pair of body parts, original kp ids in KEYPOINT_TYPE
        :return: valid inds, k x 4, each row is something like (0, 1, 3, N+j), which means main kp indices (1, 3) and
        auxiliary kp index j in image 0 is a valid triplet
        '''
        B, N = main_kp_mask.shape
        _, T = auxiliary_kp_mask.shape
        interpolation_knots = self.opts['interpolation_knots']
        N_knots = len(interpolation_knots)  # interpolated knots
        N_paths = int(T / N_knots)               # the number of body part pairs served as path for interpolation
        num_valid_kps_aux = torch.sum(auxiliary_kp_mask).long().item()
        valid_inds = torch.zeros(num_valid_kps_aux, 4).long()

        auxiliary_path_in_support = []
        for path_ind in range(N_paths):
            start_kp_id, end_kp_id = auxiliary_path[path_ind]
            start_kp_type, end_kp_type = KEYPOINT_TYPES[start_kp_id], KEYPOINT_TYPES[end_kp_id]
            start_kp_id_in_support, end_kp_id_in_support = support_kp_categories.index(start_kp_type), support_kp_categories.index(end_kp_type)
            auxiliary_path_in_support.append([start_kp_id_in_support, end_kp_id_in_support])

        cnt = 0
        for i in range(B):
            for j in range(T):
                if auxiliary_kp_mask[i, j] == 0:
                    continue
                path_ind = j // N_knots  # 0 ~ N_paths-1
                start_kp_id_in_support, end_kp_id_in_support = auxiliary_path_in_support[path_ind]
                valid_inds[cnt] = torch.tensor([i, start_kp_id_in_support, N + j, end_kp_id_in_support])

                cnt += 1

        # for i in range(B):
        #     for j in range(T):
        #         if auxiliary_kp_mask[i, j] == 0:
        #             continue
        #         path_ind = j // N_knots  # 0 ~ N_paths-1
        #         start_kp_id, end_kp_id = auxiliary_path[path_ind]
        #         start_kp_type, end_kp_type = KEYPOINT_TYPES[start_kp_id], KEYPOINT_TYPES[end_kp_id]
        #         start_kp_id_in_support, end_kp_id_in_support = support_kp_categories.index(start_kp_type), support_kp_categories.index(end_kp_type)
        #         valid_inds[cnt] = torch.tensor([i, start_kp_id_in_support, N + j, end_kp_id_in_support])
        #
        #         cnt += 1

        return valid_inds  # k x 4

    def numerical_transformation(self, lateral_out, w_computing_mode=2, offset1=1.0, offset2=0.5):
        ## transform patches_rho into positive number

        # Method 0
        if w_computing_mode == 0:
            positive_lateral_out = torch.exp(lateral_out)  # patches_rho = log(w^-1)
        # ---------------------------------------
        # Method 1
        elif w_computing_mode == 1:
            # offset1 = 1    # or 6
            # offset2 = 0.5  # or 0.5
            positive_lateral_out = offset1 * torch.sigmoid(lateral_out) + offset2  # w = a*sigmoid(patches_rho)+b
        # ---------------------------------------
        # Method 2
        elif w_computing_mode == 2:
            parameter_a = 4
            positive_lateral_out = (lateral_out + torch.sqrt(lateral_out ** 2 + parameter_a)) / 2  # f(x) = (x + sqrt(x*x + a)) / 2
            # log_w_patches = -torch.log(inv_w_patches)
        # ---------------------------------------
        # Method 3
        elif w_computing_mode == 3:
            positive_lateral_out = lateral_out ** 2  # inv_w = rho * rho
        # ---------------------------------------
        # Method 4
        elif w_computing_mode == 4:
            positive_lateral_out = nn.functional.softplus(lateral_out) + 1e-6  # f(x)= log(1+exp(x))+1e-6

        return positive_lateral_out

    def numerical_transformation_old(self, patches_rho, w_computing_mode=2, computing_inv_w=True):
        ## using patches uncertainty into consideration
        # computing_inv_w = True
        # w_computing_mode = 2
        if computing_inv_w == False:
            # ---------------------------------------
            # Method 0
            if w_computing_mode == 0:
                w_patches = torch.exp(patches_rho)  # patches_rho = log(w)
                inv_w_patches = 1 / w_patches
                # log_w_patches = patches_rho
            # ---------------------------------------
            # Method 1
            elif w_computing_mode == 1:
                offset1 = 6
                offset2 = 0.5
                w_patches = offset1 * torch.sigmoid(patches_rho) + offset2  # w = a*sigmoid(patches_rho)+b
                inv_w_patches = 1 / w_patches
                # log_w_patches = torch.log(w_patches)
            # ---------------------------------------
            # Method 2
            elif w_computing_mode == 2:
                parameter_a = 4
                w_patches = (patches_rho + torch.sqrt(patches_rho ** 2 + parameter_a)) / 2  # f(x) = (x + sqrt(x*x + a)) / 2
                inv_w_patches = 2 / (patches_rho + torch.sqrt(patches_rho ** 2 + parameter_a))  # 1 / w_patches
                # log_w_patches = -torch.log(inv_w_patches)
            # ---------------------------------------
            # Method 3
            elif w_computing_mode == 3:
                w_patches = patches_rho ** 2  # w = rho * rho
                inv_w_patches = 1 / w_patches
            # ---------------------------------------
        else:  # compute inv_w
            # ---------------------------------------
            # Method 0
            if w_computing_mode == 0:
                inv_w_patches = torch.exp(patches_rho)  # patches_rho = log(w^-1)
            # ---------------------------------------
            # Method 1
            elif w_computing_mode == 1:
                offset1 = 1
                offset2 = 0.5
                inv_w_patches = offset1 * torch.sigmoid(patches_rho) + offset2  # w = a*sigmoid(patches_rho)+b
            # ---------------------------------------
            # Method 2
            elif w_computing_mode == 2:
                parameter_a = 4
                inv_w_patches = (patches_rho + torch.sqrt(
                    patches_rho ** 2 + parameter_a)) / 2  # f(x) = (x + sqrt(x*x + a)) / 2
                # log_w_patches = -torch.log(inv_w_patches)
            # ---------------------------------------
            # Method 3
            elif w_computing_mode == 3:
                inv_w_patches = patches_rho ** 2  # inv_w = rho * rho

            w_patches = 1 / inv_w_patches
            # w_patches = 0  # not used

        return inv_w_patches, w_patches

    def get_distinctiveness_for_parts(self, p_support_lateral_out, p_query_lateral_out, support_kps, support_kp_mask, query_kps):
        computing_inv_w = True

        if computing_inv_w == True:
            inv_w_patches_s, _ = extract_representations(p_support_lateral_out, support_kps, None, context_mode='soft_fiber_bilinear')  # B1 x 1 x T
            # w_patches_s, _ = extract_representations(p_support_lateral_out.reciprocal(), support_kps, None, context_mode='soft_fiber_bilinear')  # B1 x 1 x T
            # w_patches_s = 1 / inv_w_patches_s
            w_patches_s = None
        else:
            w_patches_s, _ = extract_representations(p_support_lateral_out, support_kps, None, context_mode='soft_fiber_bilinear')  # B1 x 1 x T
            inv_w_patches_s, _ = extract_representations(p_support_lateral_out.reciprocal(), support_kps, None, context_mode='soft_fiber_bilinear')  # B1 x 1 x T
            # inv_w_patches_s = 1 / w_patches_s
        # inv_w_patches_s = inv_w_patches_s.mean(dim=0).squeeze()  # T
        inv_w_patches_s = average_representations2(inv_w_patches_s, support_kp_mask).squeeze()  # T
        # w_patches_s = average_representations2(w_patches_s, support_kp_mask).squeeze() # T

        if computing_inv_w == True:
            inv_w_patches_q, _ = extract_representations(p_query_lateral_out, query_kps, None, context_mode='soft_fiber_bilinear')  # B2 x 1 x T
            # w_patches_q, _ = extract_representations(p_query_lateral_out.reciprocal(), query_kps, None, context_mode='soft_fiber_bilinear')  # B2 x 1 x T
            # w_patches_q = 1 / inv_w_patches_q
            w_patches_q = None
        else:
            w_patches_q, _ = extract_representations(p_query_lateral_out, query_kps, None, context_mode='soft_fiber_bilinear')  # B2 x 1 x T
            inv_w_patches_q, _ = extract_representations(p_query_lateral_out.reciprocal(), query_kps, None, context_mode='soft_fiber_bilinear')  # B2 x 1 x T
            # inv_w_patches_q = 1 / w_patches_q

        B2, T = query_kps.shape[:2]
        inv_w_patches = (inv_w_patches_s.view(1, T) + inv_w_patches_q.reshape(B2, T)) / 2  # w^-1 = (w_s^-1 + w_q^-1) / 2, using broadcast, B2 x T
        # inv_w_patches = inv_w_patches_sa
        w_patches_aux = None

        return inv_w_patches, w_patches_aux

    def multiscale_regression2(self, keypoint_descriptors, query_kp_label, valid_kp_mask, keypoint_descriptors_aux, query_kp_label_aux, valid_kp_mask_aux, grid_length_list, support_kp_categories, auxiliary_paths, weight=None,  inv_w_patches_aux=None, w_patches_aux=None, inv_w_patches=None, w_patches=None):
        B2 = query_kp_label.shape[0]  # B2 x N x 2, N keypoints for each image
        N = query_kp_label.shape[1]
        num_valid_kps = torch.sum(valid_kp_mask)

        T = query_kp_label_aux.shape[1]  # B2 x T x 2, T auxiliary keypoints, T = (N_knots * N_paths)
        num_valid_kps_aux = torch.sum(valid_kp_mask_aux)

        loss_grid_class = 0
        loss_deviation = 0
        mean_predictions = 0
        mean_predictions_aux = 0

        loss_grid_class_aux = 0
        loss_deviation_aux = 0
        loss_covar_multiple_kps = 0
        loss_covar_multiple_kps_aux = 0

        # loss_deviation = torch.tensor(0.).cuda()
        # loss_deviation_aux = torch.tensor(0.).cuda()
        # loss_covar_multiple_kps = torch.tensor(0.).cuda()
        # loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()


        # ---------------find valid pairs of keypoints-------------------
        ## A) pairs in either main kps or auxiliary kps
        # valid_pair_inds = self.find_valid_pair_kps(valid_kp_mask)  # k1 x 3
        # valid_pair_inds_aux = self.find_valid_pair_kps(valid_kp_mask_aux)  # k2 x 3
        # k1 = valid_pair_inds.shape[0]
        # k2 = valid_pair_inds_aux.shape[0]
        # if k1 > 0:
        #     pair_kps_mask = torch.cat([valid_kp_mask[valid_pair_inds[:, 0], valid_pair_inds[:, 1]], valid_kp_mask[valid_pair_inds[:, 0], valid_pair_inds[:, 2]]], dim=0)
        #     pair_kps_mask = pair_kps_mask.reshape(2, k1).permute(1, 0)  # k1 x 2
        # else:
        #     loss_covar_multiple_kps = torch.tensor(0.).cuda()
        # if k2 > 0:
        #     pair_kps_mask_aux = torch.cat([valid_kp_mask_aux[valid_pair_inds_aux[:, 0], valid_pair_inds_aux[:, 1]], valid_kp_mask_aux[valid_pair_inds_aux[:, 0], valid_pair_inds_aux[:, 2]]], dim=0)
        #     pair_kps_mask_aux = pair_kps_mask_aux.reshape(2, k2).permute(1, 0)  # k2 x 2
        # else:
        #     loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()

        ## B) pairs in all kps
        # valid_kp_mask_combined = torch.cat([valid_kp_mask, valid_kp_mask_aux], dim=1)  # B X (N+T)
        # valid_pair_inds = self.find_valid_pair_kps(valid_kp_mask_combined)  # k3 x 3
        # k3 = valid_pair_inds.shape[0]
        # if k3 > 0:
        #     pair_kps_mask = torch.ones(k3, 2).long().cuda()  # k3 x 2
        #     loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()
        # else:
        #     loss_covar_multiple_kps = torch.tensor(0.).cuda()
        #     loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()

        # # C) pairs in neighboring kps
        # valid_pair_inds = self.find_valid_pair_neighbor_kps(valid_kp_mask, valid_kp_mask_aux, support_kp_categories, auxiliary_paths)  # k4 x 3
        # k4 = valid_pair_inds.shape[0]
        # if k4 > 0:
        #     pair_kps_mask = torch.ones(k4, 2).long().cuda()  # k4 x 2
        #     loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()
        #     if inv_w_patches_aux is not None:
        #         if len(inv_w_patches_aux.shape) == 1:  # only use support patches to compute inv_w, size is T
        #             if inv_w_patches is None:
        #                 inv_w_patches_ones = torch.ones(B2, N).cuda()  # set the patches rho of main trianing kps to be one
        #                 inv_w_patches_combine = torch.cat([inv_w_patches_ones, inv_w_patches_aux], dim=1)  # B2 x (N+T)
        #             else:
        #                 inv_w_patches_combine = torch.cat([inv_w_patches, inv_w_patches_aux], dim=1)  # B2 x (N+T)
        #             # k4 x 2, each row is the inv w of patches for two kps
        #             inv_w_patches_pair = torch.cat([(inv_w_patches_combine[valid_pair_inds[:, 1]]).unsqueeze(dim=1), (inv_w_patches_combine[valid_pair_inds[:, 2]]).unsqueeze(dim=1)], dim=1)  # k4 x 2
        #         elif len(inv_w_patches_aux.shape) == 2:  # use both support & query patches, size is B2 x T
        #             if inv_w_patches is None:
        #                 inv_w_patches_ones = torch.ones(B2, N).cuda()  # set the patches rho of main trianing kps to be one
        #                 inv_w_patches_combine = torch.cat([inv_w_patches_ones, inv_w_patches_aux], dim=1)  # B2 x (N+T)
        #             else:
        #                 inv_w_patches_combine = torch.cat([inv_w_patches, inv_w_patches_aux], dim=1)  # B2 x (N+T)
        #
        #             # k4 x 2, each row is the inv w of patches for two kps
        #             inv_w_patches_pair = torch.cat([(inv_w_patches_combine[valid_pair_inds[:, 0], valid_pair_inds[:, 1]]).unsqueeze(dim=1), \
        #                                             (inv_w_patches_combine[valid_pair_inds[:, 0], valid_pair_inds[:, 2]]).unsqueeze(dim=1)], dim=1)  # k4 x 2
        # else:
        #     loss_covar_multiple_kps = torch.tensor(0.).cuda()
        #     loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()

        ## D) triplets in each interpolation path
        valid_triplet_inds = self.find_valid_triplet_kps(valid_kp_mask, valid_kp_mask_aux, support_kp_categories, auxiliary_paths)  # k5 x 4
        k5 = valid_triplet_inds.shape[0]
        if k5 > 0:
            triplet_kp_mask = torch.ones(k5, 3).long().cuda()  # k5 x 3, triplet keypoints' mask
            loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()
            if inv_w_patches_aux is not  None:
                if len(inv_w_patches_aux.shape) == 1:  # only use support patches to compute inv_w, size is T
                    if inv_w_patches is None:
                        inv_w_patches_ones = torch.ones(N).cuda()  # set the patches rho of main trianing kps to be one
                        inv_w_patches_combine = torch.cat([inv_w_patches_ones, inv_w_patches_aux], dim=0)  # (N+T)
                    else:
                        inv_w_patches_combine = torch.cat([inv_w_patches, inv_w_patches_aux], dim=0)  # (N+T)
                    # inv_w_patches_combine = torch.cat([inv_w_patches, w_patches], dim=0)  # (N+T)
                    # k5 x 3, each row is the inv w of patches for two main training kps and one auxiliary kp
                    inv_w_patches_triplet = torch.cat([(inv_w_patches_combine[valid_triplet_inds[:, 1]]).unsqueeze(dim=1), (inv_w_patches_combine[valid_triplet_inds[:, 2]]).unsqueeze(dim=1), (inv_w_patches_combine[valid_triplet_inds[:, 3]]).unsqueeze(dim=1)], dim=1)  # k5 x 3
                elif len(inv_w_patches_aux.shape) == 2:  # use both support & query patches, size is B2 x T
                    if inv_w_patches is None:
                        inv_w_patches_ones = torch.ones(B2, N).cuda()  # set the patches rho of main trianing kps to be one
                        inv_w_patches_combine = torch.cat([inv_w_patches_ones, inv_w_patches_aux], dim=1)  # B2 x (N+T)
                    else:
                        inv_w_patches_combine = torch.cat([inv_w_patches, inv_w_patches_aux], dim=1)  # B2 x (N+T)
                    # k5 x 3, each row is the inv w of patches for two main training kps and one auxiliary kp
                    inv_w_patches_triplet = torch.cat([(inv_w_patches_combine[valid_triplet_inds[:, 0], valid_triplet_inds[:, 1]]).unsqueeze(dim=1), \
                                                       (inv_w_patches_combine[valid_triplet_inds[:, 0], valid_triplet_inds[:, 2]]).unsqueeze(dim=1), \
                                                       (inv_w_patches_combine[valid_triplet_inds[:, 0], valid_triplet_inds[:, 3]]).unsqueeze(dim=1)], dim=1)  # k5 x 3
        else:
            loss_covar_multiple_kps = torch.tensor(0.).cuda()
            loss_covar_multiple_kps_aux = torch.tensor(0.).cuda()


        scale_num = len(grid_length_list)
        for scale_i, grid_length in enumerate(grid_length_list):
            # ---------------main keypoints-------------------
            # compute grid groundtruth and deviation
            gridxy = (query_kp_label /2 + 0.5) * grid_length  # coordinate -1~1 --> 0~self.grid_length, B2 x N x 2
            gridxy_quantized = gridxy.long().clamp(0, grid_length - 1)  # B2 x N x 2
            # Method 1, deviation range: -1~1
            label_deviations = (gridxy - (gridxy_quantized + 0.5)) * 2  # we hope the deviation ranges -1~1, B2 x N x 2
            # Method 2, deviation range: 0~1
            # label_deviations = (gridxy - gridxy_quantized)  # we hope the deviation ranges 0~1, B2 x N x 2
            label_grids = gridxy_quantized[:, :, 1] * grid_length + gridxy_quantized[:, :, 0]  # 0 ~ grid_length * grid_length - 1, B2 x N

            one_hot_grid_label = torch.zeros(B2, N, grid_length**2).cuda()  # B2 x N x (grid_length * grid_length)
            one_hot_grid_label = one_hot_grid_label.scatter(dim=2, index=torch.unsqueeze(label_grids, dim=2), value=1)  # B2 x N x (grid_length * grid_length)

            # ---------------auxiliary keypoints-------------------
            gridxy_aux = (query_kp_label_aux / 2 + 0.5) * grid_length  # B2 x T x 2
            gridxy_quantized_aux = gridxy_aux.long().clamp(0, grid_length - 1)  # B2 x T x 2
            label_deviations_aux = (gridxy_aux - (gridxy_quantized_aux + 0.5)) * 2  # B2 x T x 2
            label_grids_aux = gridxy_quantized_aux[:, :, 1] * grid_length + gridxy_quantized_aux[:, :, 0]  # 0 ~ grid_length * grid_length - 1, B2 x T
            one_hot_grid_label_aux = torch.zeros(B2, T, grid_length**2).cuda()  # B2 x T x (grid_length * grid_length)
            one_hot_grid_label_aux = one_hot_grid_label_aux.scatter(dim=2, index=torch.unsqueeze(label_grids_aux, dim=2), value=1)  # B2 x T x (grid_length * grid_length)

            # ---------------main keypoints-------------------
            # 1) global deviation
            # predict_grids, predict_deviations, rho, _ = self.regressor[scale_i](keypoint_descriptors, training_phase=True, one_hot_grid_label=one_hot_grid_label)  # B2 x N x (grid_length ** 2), B2 x N x 2, B2 x N x 2
            # 2) local deviation
            predict_grids, predict_deviations, rho, _ = self.regressor[scale_i](keypoint_descriptors, training_phase=True, one_hot_grid_label=label_grids)  # B2 x N x (grid_length ** 2), B2 x N x 2, B2 x N x (2d)

            # ---------------auxiliary keypoints-------------------
            # 1) global deviation
            # predict_grids_aux, predict_deviations_aux, rho_aux, _ = self.regressor[scale_i](keypoint_descriptors_aux, training_phase=True, one_hot_grid_label=one_hot_grid_label_aux)
            # 2) local deviation
            predict_grids_aux, predict_deviations_aux, rho_aux, _ = self.regressor[scale_i](keypoint_descriptors_aux, training_phase=True, one_hot_grid_label=label_grids_aux)  # B2 x T x (grid_length ** 2), B2 x T x 2, B2 x T x (2d)


            # ---------------find valid pairs of keypoints-------------------
            ## A)
            # if k1 > 0:
            #     conditioned_descriptors = torch.cat([one_hot_grid_label, keypoint_descriptors], dim=2)  # B x N x (L*L+C)
            #     pair_descriptors = torch.cat([conditioned_descriptors[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], conditioned_descriptors[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1)  # k1 x (2C')
            #     pair_rho = self.covar_branch[scale_i](pair_descriptors)  # k1 x (2 x 2 x 2 x d)
            #     pair_predict_deviations = torch.cat([predict_deviations[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], predict_deviations[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1) # k1 x (2*2)
            #     pair_label_deviations = torch.cat([label_deviations[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], label_deviations[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1) # k1 x (2*2)
            #     pair_rho = pair_rho.reshape(k1, 2, 2, -1)  # k1 x 2 x 2 x 2d
            #     pair_predict_deviations = pair_predict_deviations.reshape(k1, 2, 2)
            #     pair_label_deviations = pair_label_deviations.reshape(k1, 2, 2)
            #
            # if k2 > 0:
            #     conditioned_descriptors_aux = torch.cat([one_hot_grid_label_aux, keypoint_descriptors_aux], dim=2)  # B x T x (L*L+C)
            #     pair_descriptors_aux = torch.cat([conditioned_descriptors_aux[valid_pair_inds_aux[:, 0], valid_pair_inds_aux[:, 1], :], conditioned_descriptors_aux[valid_pair_inds_aux[:, 0], valid_pair_inds_aux[:, 2], :]], dim=1)  # k2 x (2C')
            #     pair_rho_aux = self.covar_branch[scale_i](pair_descriptors_aux)  # k2 x (2 x 2 x 2 x d)
            #     pair_predict_deviations_aux = torch.cat([predict_deviations_aux[valid_pair_inds_aux[:, 0], valid_pair_inds_aux[:, 1], :], predict_deviations_aux[valid_pair_inds_aux[:, 0], valid_pair_inds_aux[:, 2], :]], dim=1) # k2 x (2*2)
            #     pair_label_deviations_aux = torch.cat([label_deviations_aux[valid_pair_inds_aux[:, 0], valid_pair_inds_aux[:, 1], :], label_deviations_aux[valid_pair_inds_aux[:, 0], valid_pair_inds_aux[:, 2], :]], dim=1) # k2 x (2*2)
            #     pair_rho_aux = pair_rho_aux.reshape(k2, 2, 2, -1)  # k2 x 2 x 2 x 2d
            #     pair_predict_deviations_aux = pair_predict_deviations_aux.reshape(k2, 2, 2)
            #     pair_label_deviations_aux = pair_label_deviations_aux.reshape(k2, 2, 2)

            ## B)
            # if k3 > 0:
            #     conditioned_descriptors = torch.cat([one_hot_grid_label, keypoint_descriptors], dim=2)  # B x N x (L*L+C)
            #     conditioned_descriptors_aux = torch.cat([one_hot_grid_label_aux, keypoint_descriptors_aux], dim=2)  # B x T x (L*L+C)
            #     descriptors_combined = torch.cat([conditioned_descriptors, conditioned_descriptors_aux], dim=1)  # B x (N+T) x (L*L+C)
            #     pair_descriptors = torch.cat([descriptors_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], descriptors_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1)  # k3 x (2C')
            #     pair_rho = self.covar_branch[scale_i](pair_descriptors)  # k3 x (2 x 2 x 2 x d)
            #     predict_deviations_combined = torch.cat([predict_deviations, predict_deviations_aux], dim=1)  # B x (N+T) x 2
            #     label_deviations_combined = torch.cat([label_deviations, label_deviations_aux], dim=1)        # B x (N+T) x 2
            #     pair_predict_deviations = torch.cat([predict_deviations_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], predict_deviations_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1) # k3 x (2*2)
            #     pair_label_deviations = torch.cat([label_deviations_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], label_deviations_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1) # k3 x (2*2)
            #     pair_rho = pair_rho.reshape(k3, 2, 2, -1)  # k3 x 2 x 2 x 2d
            #     pair_predict_deviations = pair_predict_deviations.reshape(k3, 2, 2)
            #     pair_label_deviations = pair_label_deviations.reshape(k3, 2, 2)

            ## C)
            # if k4 > 0:
            #     conditioned_descriptors = torch.cat([one_hot_grid_label, keypoint_descriptors], dim=2)  # B x N x (L*L+C)
            #     conditioned_descriptors_aux = torch.cat([one_hot_grid_label_aux, keypoint_descriptors_aux], dim=2)  # B x T x (L*L+C)
            #     descriptors_combined = torch.cat([conditioned_descriptors, conditioned_descriptors_aux], dim=1)  # B x (N+T) x (L*L+C)
            #     pair_descriptors = torch.cat([descriptors_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], descriptors_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1)  # k4 x (2C')
            #     pair_rho = self.covar_branch[scale_i](pair_descriptors)  # k4 x (2 x 2 x 2 x d)
            #     predict_deviations_combined = torch.cat([predict_deviations, predict_deviations_aux], dim=1)  # B x (N+T) x 2
            #     label_deviations_combined = torch.cat([label_deviations, label_deviations_aux], dim=1)        # B x (N+T) x 2
            #     pair_predict_deviations = torch.cat([predict_deviations_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], predict_deviations_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1) # k4 x (2*2)
            #     pair_label_deviations = torch.cat([label_deviations_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 1], :], label_deviations_combined[valid_pair_inds[:, 0], valid_pair_inds[:, 2], :]], dim=1) # k4 x (2*2)
            #     pair_rho = pair_rho.reshape(k4, 2, 2, -1)  # k4 x 2 x 2 x 2d
            #     pair_predict_deviations = pair_predict_deviations.reshape(k4, 2, 2)
            #     pair_label_deviations = pair_label_deviations.reshape(k4, 2, 2)

            ## D)
            if k5 > 0:
                conditioned_descriptors = torch.cat([one_hot_grid_label, keypoint_descriptors], dim=2)  # B x N x (L*L+C)
                conditioned_descriptors_aux = torch.cat([one_hot_grid_label_aux, keypoint_descriptors_aux], dim=2)  # B x T x (L*L+C)
                descriptors_combined = torch.cat([conditioned_descriptors, conditioned_descriptors_aux], dim=1)  # B x (N+T) x (L*L+C)
                triplet_descriptors = torch.cat([descriptors_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 1], :], descriptors_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 2], :], \
                                    descriptors_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 3], :]], dim=1)  # k5 x (3C')
                triplet_rho = self.covar_branch[scale_i](triplet_descriptors)  # k5 x (3 x 3 x 2 x d)
                predict_deviations_combined = torch.cat([predict_deviations, predict_deviations_aux], dim=1)  # B x (N+T) x 2
                label_deviations_combined = torch.cat([label_deviations, label_deviations_aux], dim=1)        # B x (N+T) x 2
                triplet_predict_deviations = torch.cat([predict_deviations_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 1], :], predict_deviations_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 2], :], \
                                    predict_deviations_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 3], :]], dim=1) # k5 x (3*2)
                triplet_label_deviations = torch.cat([label_deviations_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 1], :], label_deviations_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 2], :], \
                                    label_deviations_combined[valid_triplet_inds[:, 0], valid_triplet_inds[:, 3], :]], dim=1) # k5 x (3*2)
                triplet_rho = triplet_rho.reshape(k5, 3, 3, -1)  # k5 x 3 x 3 x 2d
                triplet_predict_deviations = triplet_predict_deviations.reshape(k5, 3, 2)
                triplet_label_deviations = triplet_label_deviations.reshape(k5, 3, 2)



            # print_weights(valid_kp_mask)
            # print_weights(valid_pair_inds)
            # print_weights(valid_kp_mask_aux)
            # print_weights(valid_pair_inds_aux)



            # ---------------main keypoints-------------------
            # compute grid classification loss and deviation loss
            predict_grids2 = predict_grids.view(B2 * N, -1)   # (B2 * N) * (grid_length * grid_length)
            label_grids2 = label_grids
            for i in range(B2):
                for j in range(N):
                    if valid_kp_mask[i, j] < 1:
                        label_grids2[i, j] = -1  # set ignore index for nllloss
            label_grids2 = label_grids2.view(-1)  # (B2 * N)

            # ---------------auxiliary keypoints-------------------
            predict_grids_aux2 = predict_grids_aux.view(B2 * T, -1)   # (B2 * T) * (grid_length * grid_length)
            label_grids_aux2 = label_grids_aux
            for i in range(B2):
                for j in range(T):
                    if valid_kp_mask_aux[i, j] < 1:
                        label_grids_aux2[i, j] = -1  # set ignore index for nllloss
            label_grids_aux2 = label_grids_aux2.view(-1)  # (B2 * T)

            if weight is None:
                square_root_for_w = True

                # A) main training kps, grid classification
                if inv_w_patches is None:
                    loss_grid_class += self.loss_fun_nll(predict_grids2, label_grids2)
                else:
                    # loss_grid_class += self.loss_fun_nll(predict_grids2, label_grids2)

                    # # using patches uncertainty into consideration
                    if square_root_for_w == False:
                        if len(inv_w_patches.shape) == 1:  # only use support patches to compute inv_w, size is N
                            instance_weight = inv_w_patches.repeat(B2, 1).reshape(-1)  # (B2*N)
                        elif len(inv_w_patches.shape) == 2:  # use both support & query patches, size is B2 x N
                            instance_weight = inv_w_patches.reshape(-1)  # (B2*N)
                    else:
                        if len(inv_w_patches.shape) == 1:  # only use support patches to compute inv_w, size is N
                            instance_weight = torch.sqrt(inv_w_patches+1e-9).repeat(B2, 1).reshape(-1)
                        elif len(inv_w_patches.shape) == 2:  # use both support & query patches, size is B2 x N
                            instance_weight = torch.sqrt(inv_w_patches+1e-9).reshape(-1)

                    loss_grid_class += instance_weighted_nllloss(predict_grids2, label_grids2, instance_weight=instance_weight, ignore_index=-1)  # weight: (B2*N)

                # B) auxiliary kps, grid classification
                if inv_w_patches_aux is None:
                    loss_grid_class_aux += self.loss_fun_nll(predict_grids_aux2, label_grids_aux2)
                else:
                    # loss_grid_class_aux += self.loss_fun_nll(predict_grids_aux2, label_grids_aux2)

                    # # using patches uncertainty into consideration
                    # ---------------------------------------
                    if square_root_for_w == False:
                        if len(inv_w_patches_aux.shape) == 1:  # only use support patches to compute inv_w, size is T
                            instance_weight_aux = inv_w_patches_aux.repeat(B2, 1).reshape(-1)  # (B2*T)
                        elif len(inv_w_patches_aux.shape) == 2:  # use both support & query patches, size is B2 x T
                            instance_weight_aux = inv_w_patches_aux.reshape(-1)  # (B2*T)
                    else:
                        if len(inv_w_patches_aux.shape) == 1:  # only use support patches to compute inv_w, size is T
                            instance_weight_aux = torch.sqrt(inv_w_patches_aux).repeat(B2, 1).reshape(-1)
                        elif len(inv_w_patches_aux.shape) == 2:  # use both support & query patches, size is B2 x T
                            instance_weight_aux = torch.sqrt(inv_w_patches_aux).reshape(-1)

                    loss_grid_class_aux += instance_weighted_nllloss(predict_grids_aux2, label_grids_aux2, instance_weight=instance_weight_aux, ignore_index=-1)  # weight: T
                    # compute penatly for weights
                    # penalty = torch.sum(log_w_patches.repeat(B2, 1) * valid_kp_mask_aux)  # use log(w) as penalty
                    # if num_valid_kps_aux > 0:
                    #     penalty /= num_valid_kps_aux
                    # # print(penalty)
                    # loss_grid_class_aux += penalty
                    # ---------------------------------------
            else:
                loss_grid_class += self.loss_fun_nll(predict_grids2, label_grids2)
                instance_weight = weight.repeat(B2, 1).reshape(-1)  # construct instance_weights which has (B*N) elements
                loss_grid_class_aux += instance_weighted_nllloss(predict_grids_aux2, label_grids_aux2, instance_weight=instance_weight, ignore_index=-1)  # weight: N

            # ------------------
            if weight is None:
                loss_fun_mode = 0
                penalty_mode = 0  # 0 or 2 are better, namely log(det(W^-1)) or 1/ W^-1 - 1
                beta = 1.0
                beta_loc_uc = 1.0

                # A) main training kps, deviation regression
                if inv_w_patches is None:
                    # covariance for each keypoint
                    loss_deviation += masked_nll_gaussian_covar(predict_deviations, label_deviations, rho, valid_kp_mask)
                else:
                    # covariance for each keypoint
                    loss_deviation += masked_nll_gaussian_covar(predict_deviations, label_deviations, rho, valid_kp_mask, patches_rho=inv_w_patches, loss_fun_mode=loss_fun_mode, penalty_mode=penalty_mode, beta=beta, beta_loc_uc=beta_loc_uc)

                # B) auxiliary kps, deviation regression
                if inv_w_patches_aux is None:
                    # independent variance
                    # loss_deviation += masked_nll_gaussian(predict_deviations, label_deviations, rho, valid_kp_mask.view(B2, N, 1))
                    # loss_deviation += masked_nll_laplacian(predict_deviations, label_deviations, rho, valid_kp_mask.view(B2, N, 1))

                    # covariance for each keypoint
                    loss_deviation_aux += masked_nll_gaussian_covar(predict_deviations_aux, label_deviations_aux, rho_aux, valid_kp_mask_aux)

                    # covariance for all keypoints
                    # loss_deviation += masked_nll_gaussian_covar2(predict_deviations, label_deviations, rho, valid_kp_mask, method=1)

                    # ---------------covariance for valid pairs of keypoints-------------------
                    ## A) pairs in either main kps or auxiliary kps
                    # if k1 > 0:
                    #     loss_covar_multiple_kps += masked_nll_gaussian_covar2(pair_predict_deviations, pair_label_deviations, pair_rho, pair_kps_mask, covar_method=0)
                    # if k2 > 0:
                    #     loss_covar_multiple_kps_aux += masked_nll_gaussian_covar2(pair_predict_deviations_aux, pair_label_deviations_aux, pair_rho_aux, pair_kps_mask_aux, covar_method=0)
                    ## B) pairs in all kps
                    # if k3 > 0:
                    #     loss_covar_multiple_kps += masked_nll_gaussian_covar2(pair_predict_deviations, pair_label_deviations, pair_rho, pair_kps_mask, covar_method=0)
                    ## C) pairs in neighboring kps
                    # if k4 > 0:
                    #     loss_covar_multiple_kps += masked_nll_gaussian_covar2(pair_predict_deviations, pair_label_deviations, pair_rho, pair_kps_mask, covar_method=0)
                    ## D) triplets in each interpolation path
                    if k5 > 0:
                        loss_covar_multiple_kps += masked_nll_gaussian_covar2(triplet_predict_deviations, triplet_label_deviations, triplet_rho, triplet_kp_mask, covar_method=0)

                    # # no variance
                    # loss_deviation += masked_l1_loss(predict_deviations, label_deviations, valid_kp_mask.view(B2, N, 1))
                    # loss_deviation_aux += masked_l1_loss(predict_deviations_aux, label_deviations_aux, valid_kp_mask_aux.view(B2, T, 1))
                    # loss_deviation += masked_mse_loss(predict_deviations, label_deviations, valid_kp_mask.view(B2, N, 1))
                    # loss_deviation_aux += masked_mse_loss(predict_deviations_aux, label_deviations_aux, valid_kp_mask_aux.view(B2, T, 1))

                else:
                    # # no variance
                    # loss_deviation += masked_l1_loss(predict_deviations, label_deviations, valid_kp_mask.view(B2, N, 1))
                    # loss_deviation_aux += masked_l1_loss(predict_deviations_aux, label_deviations_aux, valid_kp_mask_aux.view(B2, N, 1))
                    # loss_deviation += masked_mse_loss(predict_deviations, label_deviations, valid_kp_mask.view(B2, N, 1))
                    # loss_deviation_aux += masked_mse_loss(predict_deviations_aux, label_deviations_aux, valid_kp_mask_aux.view(B2, N, 1))


                    # covariance for each keypoint
                    loss_deviation_aux += masked_nll_gaussian_covar(predict_deviations_aux, label_deviations_aux, rho_aux, valid_kp_mask_aux, patches_rho=inv_w_patches_aux, loss_fun_mode=loss_fun_mode, penalty_mode=penalty_mode, beta=beta, beta_loc_uc=beta_loc_uc)


                    # ---------------covariance for valid pairs of keypoints-------------------
                    ## A) pairs in either main kps or auxiliary kps
                    # if k1 > 0:
                    #     loss_covar_multiple_kps += masked_nll_gaussian_covar2(pair_predict_deviations, pair_label_deviations, pair_rho, pair_kps_mask, covar_method=0)
                    # if k2 > 0:
                    #     loss_covar_multiple_kps_aux += masked_nll_gaussian_covar2(pair_predict_deviations_aux, pair_label_deviations_aux, pair_rho_aux, pair_kps_mask_aux, covar_method=0)
                    ## B) pairs in all kps
                    # if k3 > 0:
                    #     loss_covar_multiple_kps += masked_nll_gaussian_covar2(pair_predict_deviations, pair_label_deviations, pair_rho, pair_kps_mask, covar_method=0)
                    ## C) pairs in neighboring kps
                    # if k4 > 0:
                    #     loss_covar_multiple_kps += masked_nll_gaussian_covar2(pair_predict_deviations, pair_label_deviations, pair_rho, pair_kps_mask, covar_method=0, patches_rho=inv_w_patches_pair)
                    ## D) triplets in each interpolation path
                    if k5 > 0:
                        loss_covar_multiple_kps += masked_nll_gaussian_covar2(triplet_predict_deviations, triplet_label_deviations, triplet_rho, triplet_kp_mask, covar_method=0, patches_rho=inv_w_patches_triplet, loss_fun_mode=loss_fun_mode, penalty_mode=penalty_mode, beta=beta, beta_loc_uc=beta_loc_uc)

                    # using patches uncertainty into consideration
                    # loss_deviation += masked_nll_laplacian(predict_deviations, label_deviations, patches_rho.view(1, N, 1), valid_kp_mask.view(B2, N, 1), beta=0.5, computing_mode=0, offset1=6, offset2=0.5)
                    # loss_deviation += masked_nll_gaussian(predict_deviations, label_deviations, patches_rho.view(1, N, 1), valid_kp_mask.view(B2, N, 1), beta=0.5, computing_mode=0, offset1=6, offset2=0.5)
                    # loss_deviation += masked_nll_laplacian2(predict_deviations, label_deviations, rho, patches_rho.view(1, N, 1), valid_kp_mask.view(B2, N, 1), beta=0.5, gamma=0.98)
                    # loss_deviation += masked_nll_gaussian2(predict_deviations, label_deviations, rho, patches_rho.view(1, N, 1), valid_kp_mask.view(B2, N, 1), beta=0.5, gamma=0.95)
            else:
                # weight: N
                combine_mask_weight = (valid_kp_mask * weight.reshape(1, -1)).view(B2, N, 1)
                # loss_deviation += masked_nll_gaussian(predict_deviations, label_deviations, rho, combine_mask_weight)
                # loss_deviation += masked_nll_laplacian(predict_deviations, label_deviations, rho, combine_mask_weight)
                # loss_deviation += masked_mse_loss(predict_deviations, label_deviations, combine_mask_weight)
                loss_deviation += masked_l1_loss(predict_deviations, label_deviations, combine_mask_weight)

            # ------------------
            # type2, transform to original location
            # mean_predictions += (((gridxy_quantized + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2

            # compute predicted keypoint locations using grids and deviations
            out_predict_grids = torch.max(predict_grids, dim=2)[1]  # B2 x N, 0 ~ grid_length * grid_length - 1
            out_predict_gridxy = torch.FloatTensor(B2, N, 2).cuda()
            out_predict_gridxy[:, :, 0] = out_predict_grids % grid_length  # grid x
            out_predict_gridxy[:, :, 1] = out_predict_grids // grid_length # grid y

            out_predict_grids_aux = torch.max(predict_grids_aux, dim=2)[1]  # B2 x T, 0 ~ grid_length * grid_length - 1
            out_predict_gridxy_aux = torch.FloatTensor(B2, T, 2).cuda()
            out_predict_gridxy_aux[:, :, 0] = out_predict_grids_aux % grid_length  # grid x
            out_predict_gridxy_aux[:, :, 1] = out_predict_grids_aux // grid_length  # grid y

            # Method 1, deviation range: -1~1
            predictions = (((out_predict_gridxy + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
            predictions_aux = (((out_predict_gridxy_aux + 0.5) + predict_deviations_aux / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
            # Method 2, deviation range: 0~1
            # predictions = ((predict_gridxy + predict_deviations) / grid_length - 0.5) * 2
            mean_predictions += predictions
            mean_predictions_aux += predictions_aux
            # ------------------

        # ------------------
        # type2, transform to original location
        mean_predictions /= scale_num
        mean_predictions_aux /= scale_num
        # mean_predictions /= scale_num
        # query_labels2 = query_labels * valid_kp_mask.view(B2, N, 1)
        # mean_predictions2 = mean_predictions * valid_kp_mask.view(B2, N, 1)
        # loss_deviation  = self.loss_fun_mse(mean_predictions2, query_labels2)
        # loss_deviation = loss_deviation / num_valid_kps * 15
        # ------------------

        loss_grid_class /= scale_num  # multi-scale
        loss_grid_class_aux /= scale_num
        # ------------------
        # type1, direct regressing deviation
        loss_deviation  /= scale_num  # multi-scale
        loss_deviation_aux /= scale_num


        loss_covar_multiple_kps /= scale_num
        loss_covar_multiple_kps_aux /= scale_num

        # if weight is None:
        #     num_valid_kps = torch.sum(valid_kp_mask).item()
        # else:
        #     num_valid_kps = torch.sum(valid_kp_mask * weight.reshape(1, -1)).item()
        # if num_valid_kps == 0:  # for symmetry case, it may not have the valid symmetric keypoints
        #     pass  # no need to divide symmetry_num_valid_kps since all are zero
        # else:
        #     loss_deviation = loss_deviation / num_valid_kps
        # ------------------

        # loss = loss_grid_class  + loss_deviation
        # print('loss_d: ', loss_deviation.cpu().detach().numpy())

        return loss_grid_class, loss_deviation, loss_grid_class_aux, loss_deviation_aux, loss_covar_multiple_kps, loss_covar_multiple_kps_aux, mean_predictions, mean_predictions_aux

    def load_model(self):
        if self.opts['load_trained_model']:
            postfix = self.opts['load_model_postfix']
            load_model_root = self.opts['load_model_root']
            if os.path.exists(load_model_root + '/encoder' + postfix) == False:
                print("Error in load_model: model loading path is not found!")
                exit(0)
            self.encoder.load_state_dict(torch.load(load_model_root + '/encoder' + postfix))
            self.descriptor_net.load_state_dict(torch.load(load_model_root + '/descriptor_net' + postfix))
            self.regressor.load_state_dict(torch.load(load_model_root + '/regressor' + postfix))
            # if self.opts['use_domain_confusion']:
            #     self.auxiliary_classifier.load_state_dict(torch.load(load_model_root + '/classifier' + postfix))
            #     self.auxiliary_classifier_kp.load_state_dict(torch.load(load_model_root + '/classifier_kp' + postfix))
            # if self.opts['use_auxiliary_regressor'] == True:
            #     self.auxiliary_regressor.load_state_dict(torch.load(load_model_root + '/auxiliary_regressor' + postfix))
            # if self.opts['use_pum']:
            #     self.pum.load_state_dict(torch.load(load_model_root + '/pum' + postfix))
            self.covar_branch.load_state_dict(torch.load(load_model_root + '/covar_branch' + postfix))
            # if self.opts['offset_learning']:
            #     self.offsetnet.load_state_dict(torch.load(load_model_root + '/offsetnet' + postfix))

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.descriptor_net = self.descriptor_net.cuda()
            self.regressor = self.regressor.cuda()
            # if self.opts['use_domain_confusion']:
            #     self.auxiliary_classifier = self.auxiliary_classifier.cuda()
            #     self.auxiliary_classifier_kp = self.auxiliary_classifier_kp.cuda()
            # if self.opts['use_auxiliary_regressor'] == True:
            #     self.auxiliary_regressor = self.auxiliary_regressor.cuda()
            # if self.opts['use_pum']:
            #     self.pum = self.pum.cuda()
            self.covar_branch = self.covar_branch.cuda()
            # if self.opts['offset_learning']:
            #     self.offsetnet = self.offsetnet.cuda()

            # use data parallel
            if True == self.opts['data_parallel']:
                self.encoder = nn.DataParallel(self.encoder, device_ids=(0, 1))
                self.descriptor_net = nn.DataParallel(self.descriptor_net, device_ids=(0, 1))
                # self.regressor = nn.DataParallel(self.regressor, device_ids=(0, 1))
                for i in range(len(self.grid_length)):
                    self.regressor[i] = nn.DataParallel(self.regressor[i], device_ids=(0, 1))
                    self.covar_branch[i] = nn.DataParallel(self.covar_branch[i], device_ids=(0, 1))
                # if self.opts['use_pum']:
                #     self.pum = nn.DataParallel(self.pum, device_ids=(0, 1))
                # if self.opts['offset_learning']:
                #     self.offsetnet = nn.DataParallel(self.offsetnet, device_ids=(0, 1))

    def save_model(self):
        if self.opts['save_model']:
            postfix = self.opts['save_model_postfix']
            save_model_root = self.opts['save_model_root']
            if False == self.opts['data_parallel']:
                if ('/g/data' in save_model_root) or ('/scratch' in save_model_root):  # it is used in remote server
                    torch.save(self.encoder.state_dict(), save_model_root + '/encoder' + postfix) # ,_use_new_zipfile_serialization=False)
                    torch.save(self.descriptor_net.state_dict(), save_model_root + '/descriptor_net' + postfix) # ,_use_new_zipfile_serialization=False)
                    torch.save(self.regressor.state_dict(), save_model_root + '/regressor' + postfix) # ,_use_new_zipfile_serialization=False)
                    # if self.opts['use_domain_confusion']:
                    #     torch.save(self.auxiliary_classifier.state_dict(), save_model_root + '/classifier' + postfix) # ,_use_new_zipfile_serialization=False)
                    #     torch.save(self.auxiliary_classifier_kp.state_dict(), save_model_root + '/classifier_kp' + postfix) # ,_use_new_zipfile_serialization=False)
                    # if self.opts['use_auxiliary_regressor']:
                    #     torch.save(self.auxiliary_regressor.state_dict(), save_model_root + '/auxiliary_regressor' + postfix) # , _use_new_zipfile_serialization=False)
                    # if self.opts['use_pum'] == True:
                    #     torch.save(self.pum.state_dict(), save_model_root + '/pum' + postfix) # , _use_new_zipfile_serialization=False)
                    torch.save(self.covar_branch.state_dict(), save_model_root + '/covar_branch' + postfix) # , _use_new_zipfile_serialization=False)
                    # if self.opts['offset_learning'] == True:
                    #     torch.save(self.offsetnet.state_dict(), save_model_root + '/offsetnet' + postfix) # , _use_new_zipfile_serialization=False)

                else:  # used in local computer
                    torch.save(self.encoder.state_dict(), save_model_root + '/encoder' + postfix)
                    torch.save(self.descriptor_net.state_dict(), save_model_root + '/descriptor_net' + postfix)
                    torch.save(self.regressor.state_dict(), save_model_root + '/regressor' + postfix)
                    # if self.opts['use_domain_confusion']:
                    #     torch.save(self.auxiliary_classifier.state_dict(), save_model_root + '/classifier' + postfix)
                    #     torch.save(self.auxiliary_classifier_kp.state_dict(), save_model_root + '/classifier_kp' + postfix)
                    # if self.opts['use_auxiliary_regressor']:
                    #     torch.save(self.auxiliary_regressor.state_dict(), save_model_root + '/auxiliary_regressor' + postfix)
                    # if self.opts['use_pum'] == True:
                    #     torch.save(self.pum.state_dict(), save_model_root + '/pum' + postfix)
                    torch.save(self.covar_branch.state_dict(), save_model_root + '/covar_branch' + postfix)
                    # if self.opts['offset_learning'] == True:
                    #     torch.save(self.offsetnet.state_dict(), save_model_root + '/offsetnet' + postfix)
            else:  # save models which use data parallel
                if ('/g/data' in save_model_root) or ('/scratch' in save_model_root):  # it is used in remote server
                    torch.save(self.encoder.module.state_dict(), save_model_root + '/encoder' + postfix) # ,_use_new_zipfile_serialization=False)
                    torch.save(self.descriptor_net.module.state_dict(), save_model_root + '/descriptor_net' + postfix) # ,_use_new_zipfile_serialization=False)
                    modules = []
                    covar_branch = []
                    for i in range(len(self.grid_length)):
                        modules += [self.regressor[i].module]  # retrieve net from class DataParallel
                        covar_branch += [self.covar_branch[i].module]
                    torch.save(nn.ModuleList(modules).state_dict(), save_model_root + '/regressor' + postfix) # ,_use_new_zipfile_serialization=False)
                    torch.save(nn.ModuleList(covar_branch).state_dict(), save_model_root + '/covar_branch' + postfix) # ,_use_new_zipfile_serialization=False)
                    # if self.opts['use_pum'] == True:
                    #     torch.save(self.pum.module.state_dict(), save_model_root + '/pum' + postfix) # , _use_new_zipfile_serialization=False)
                    # if self.opts['offset_learning'] == True:
                    #     torch.save(self.offsetnet.state_dict(), save_model_root + '/offsetnet' + postfix) # , _use_new_zipfile_serialization=False)
                else:  # used in local computer
                    torch.save(self.encoder.module.state_dict(), save_model_root + '/encoder' + postfix)
                    torch.save(self.descriptor_net.module.state_dict(), save_model_root + '/descriptor_net' + postfix)
                    modules = []
                    covar_branch = []
                    for i in range(len(self.grid_length)):
                        modules += [self.regressor[i].module]  # retrieve net from class DataParallel
                        covar_branch += [self.covar_branch[i].module]
                    torch.save(nn.ModuleList(modules).state_dict(), save_model_root + '/regressor' + postfix)
                    torch.save(nn.ModuleList(covar_branch).state_dict(), save_model_root + '/covar_branch' + postfix)
                    # if self.opts['use_pum'] == True:
                    #     torch.save(self.pum.module.state_dict(), save_model_root + '/pum' + postfix)
                    # if self.opts['offset_learning'] == True:
                    #     torch.save(self.offsetnet.state_dict(), save_model_root + '/offsetnet' + postfix)

    def optimizer_init(self, lr = 0.0001, lr_auxiliary = 0.0001, weight_decay = 0, optimization_algorithm = 'SGD'):
        # lr = 0.001
        if optimization_algorithm == 'SGD':
            self.optimizer1 = optim.SGD(self.encoder.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            self.optimizer2 = optim.SGD(self.descriptor_net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            self.optimizer3 = optim.SGD(self.regressor.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            self.optimizer8 = optim.SGD(self.covar_branch.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            # if self.opts['use_domain_confusion']:
            #     self.optimizer4 = optim.SGD(self.auxiliary_classifier.parameters(), lr=lr_auxiliary, momentum=0.9, weight_decay=weight_decay)
            #     self.optimizer5 = optim.SGD(self.auxiliary_classifier_kp.parameters(), lr=lr_auxiliary, momentum=0.9, weight_decay=weight_decay)

        elif optimization_algorithm == 'Adam':
            self.optimizer1 = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            self.optimizer2 = optim.Adam(self.descriptor_net.parameters(), lr=lr, weight_decay=weight_decay)
            self.optimizer3 = optim.Adam(self.regressor.parameters(), lr=lr, weight_decay=weight_decay)
            # if self.opts['use_domain_confusion']:
            #     self.optimizer4 = optim.Adam(self.auxiliary_classifier.parameters(), lr=lr_auxiliary, weight_decay=weight_decay)
            #     self.optimizer5 = optim.Adam(self.auxiliary_classifier_kp.parameters(), lr=lr_auxiliary, weight_decay=weight_decay)
            # if self.opts['use_auxiliary_regressor']:
            #     self.optimizer6 = optim.Adam(self.auxiliary_regressor.parameters(), lr=lr, weight_decay=weight_decay)
            # if self.opts['use_pum']:
            #     self.optimizer7 = optim.Adam(self.pum.parameters(), lr=lr, weight_decay=weight_decay)
            self.optimizer8 = optim.Adam(self.covar_branch.parameters(), lr=lr, weight_decay=weight_decay)
            # if self.opts['offset_learning']:
            #     self.optimizer9 = optim.Adam(self.offsetnet.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimization_algorithm == 'Adagrad':
            self.optimizer1 = optim.Adagrad(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            self.optimizer2 = optim.Adagrad(self.descriptor_net.parameters(), lr=lr, weight_decay=weight_decay)
            self.optimizer3 = optim.Adagrad(self.regressor.parameters(), lr=lr, weight_decay=weight_decay)
            if self.opts['use_domain_confusion']:
                self.optimizer4 = optim.Adagrad(self.auxiliary_classifier.parameters(), lr=lr_auxiliary, weight_decay=weight_decay)
                self.optimizer5 = optim.Adagrad(self.auxiliary_classifier_kp.parameters(), lr=lr_auxiliary, weight_decay=weight_decay)

        elif optimization_algorithm == 'RMSprop':
            self.optimizer1 = optim.RMSprop(self.encoder.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay)
            self.optimizer2 = optim.RMSprop(self.descriptor_net.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay)
            self.optimizer3 = optim.RMSprop(self.regressor.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay)
            if self.opts['use_domain_confusion']:
                self.optimizer4 = optim.RMSprop(self.auxiliary_classifier.parameters(), lr=lr_auxiliary, alpha=0.99, eps=1e-08, weight_decay=weight_decay)
                self.optimizer5 = optim.RMSprop(self.auxiliary_classifier_kp.parameters(), lr=lr_auxiliary, alpha=0.99, eps=1e-08, weight_decay=weight_decay)

        self.lr_scheduler_init()

    def lr_scheduler_init(self):
        self.lr_schedule_step_size = 20000
        decay_rate = 0.6

        self.lr_scheduler1 = optim.lr_scheduler.StepLR(self.optimizer1, self.lr_schedule_step_size, decay_rate)
        self.lr_scheduler2 = optim.lr_scheduler.StepLR(self.optimizer2, self.lr_schedule_step_size, decay_rate)
        self.lr_scheduler3 = optim.lr_scheduler.StepLR(self.optimizer3, self.lr_schedule_step_size, decay_rate)
        # self.lr_scheduler4 = optim.lr_scheduler.StepLR(self.optimizer4, self.lr_schedule_step_size, decay_rate)
        # self.lr_scheduler5 = optim.lr_scheduler.StepLR(self.optimizer5, self.lr_schedule_step_size, decay_rate)
        self.lr_scheduler8 = optim.lr_scheduler.StepLR(self.optimizer8, self.lr_schedule_step_size, decay_rate)

    def optimizer_step(self, loss):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        # if self.opts['use_domain_confusion']:
        #     self.optimizer4.zero_grad()
        #     self.optimizer5.zero_grad()
        # if self.opts['use_auxiliary_regressor']:
        #     self.optimizer6.zero_grad()
        # if self.opts['use_pum']:
        #     self.optimizer7.zero_grad()
        self.optimizer8.zero_grad()
        # if self.opts['offset_learning']:
        #     self.optimizer9.zero_grad()
        loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer3.step()
        # if self.opts['use_domain_confusion']:
        #     self.optimizer4.step()
        #     self.optimizer5.step()
        # if self.opts['use_auxiliary_regressor']:
        #     self.optimizer6.step()
        # if self.opts['use_pum']:
        #     self.optimizer7.step()
        self.optimizer8.step()
        # if self.opts['offset_learning']:
        #     self.optimizer9.step()

    def lr_scheduler_step(self, episode_i):
        self.lr_scheduler1.step(episode_i)
        self.lr_scheduler2.step(episode_i)
        self.lr_scheduler3.step(episode_i)
        # self.lr_scheduler4.step(episode_i)
        # self.lr_scheduler5.step(episode_i)
        self.lr_scheduler8.step(episode_i)

        ## print learning rate for debug
        # for param_group in self.optimizer1.param_groups:
        #     print(param_group['lr'])
        #     break
        # for param_group in self.optimizer2.param_groups:
        #     print(param_group['lr'])
        #     break
        # for param_group in self.optimizer3.param_groups:
        #     print(param_group['lr'])
        #     break

        if episode_i % self.lr_schedule_step_size == 0:
            print('lr adjusted: ', self.lr_scheduler1.get_last_lr())

    def set_eval_status(self):
        self.encoder.eval()
        self.descriptor_net.eval()
        self.regressor.eval()
        self.covar_branch.eval()

    def set_train_status(self):
        self.encoder.train()
        self.descriptor_net.train()
        self.regressor.train()
        self.covar_branch.train()

    def compute_similarity_loss(self, normed_similarity_map, query_labels, valid_kp_mask):
        '''
        :param normed_similarity_map: B2 x N x H x W
        :param query_labels: B2 x N x 2
        :param valid_kp_mask: B2 x N
        :return: similarity_loss
        '''
        B2, N, _, W = normed_similarity_map.shape
        hard_query_labels = (query_labels / 2 + 0.5).mul(W-1).round().long().clamp(0, W-1)
        pos_similarity_loss = 0
        neg_similarity_loss = 0
        for i in range(B2):
            for j in range(N):
                if valid_kp_mask[i, j] > 0:  # valid keypoint
                    x, y = hard_query_labels[i, j]
                    pos_similarity_loss += (1 - normed_similarity_map[i, j, y, x])  # 1 - s
                    neg_similarity_loss += (torch.sum(normed_similarity_map[i, j]) - normed_similarity_map[i, j, y, x]) / (W**2 - 1)  # s

        num_valid_kps = torch.sum(valid_kp_mask)
        total_similarity_loss = pos_similarity_loss  # (pos_similarity_loss + neg_similarity_loss)
        if num_valid_kps > 0:
            total_similarity_loss /= num_valid_kps
        else:
            total_similarity_loss = torch.tensor(0., requires_grad=True).cuda()  # set 0 in cuda device in order to be compatible with subsequent code

        return total_similarity_loss

    # @torch.no_grad()
    def validate(self, episode_generator, num_test_episodes=1000, eval_method='method1', using_crop=True):
        torch.set_grad_enabled(False)  # disable grad computation
        if episode_generator == None:
            return

        print('==============testing start==============')
        square_image_length = self.opts['square_image_length']  # 368
        pck_thresh_bbx =  np.array([0.1, 0.15]) # np.linspace(0, 1, 101)
        # pck_thresh_img = np.array([20.0 / 368, 40.0 / 368])  # 0.06 * 384 = 23.04 pixels (23 pixels)
        pck_thresh_img = np.array([0.06, 0.1])  # 0.06 * 384 = 23.04 pixels (23 pixels)
        pck_thresh_type = 'bbx'  # 'bbx' or 'img'
        if pck_thresh_type == 'bbx':  # == 'bbx'
            pck_thresh = pck_thresh_bbx
        else:  # == 'img'
            pck_thresh = pck_thresh_img

        ks_thresh = np.array([0.5, 0.75])
        tps, fps = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]
        tps2, fps2 = [[] for _ in range(len(ks_thresh))], [[] for _ in range(len(ks_thresh))]

        tps3, fps3 = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]
        # tps4, fps4 = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]

        acc_list = [[] for _ in range(len(pck_thresh))]

        #----------------------
        # used to compute relationship between normalized distance error and localization uncertainty
        # d_increment = 0.05
        # d_norm_max = 0.3  # 0~d_norm_max
        # N_bins = d_norm_max / d_increment  # 3/0.05=5.9999999
        # if 1+int(N_bins)-N_bins <= 1e-9:
        #     N_bins = 1+int(N_bins)
        # else:
        #     N_bins = int(N_bins)
        # d_uc_bins = np.zeros((N_bins, 4))  # each row store summation of d_dorm_i, uc_loc_i, uc_sd_i (optional), count
        #----------------------


        if self.opts['set_eval']:
            self.set_eval_status()
        episode_i = 0
        sample_failure_cnt = 0
        # ---
        finetuning_steps = self.opts['finetuning_steps']
        if finetuning_steps > 0:
            original_params_dict = self.copy_original_params()
        # ---
        if self.opts['use_body_part_protos'] and self.opts['eval_with_body_part_protos']:
            self.load_proto_memory()
            body_part_proto = self.memory['proto']  # C x N
            proto_mask = self.memory['proto_mask']  # N
        #---
        while episode_i < num_test_episodes:
            # roll-out an episode
            if (False == episode_generator.episode_next()):
                sample_failure_cnt += 1
                if sample_failure_cnt % 500 == 0:
                    print('sample failure times: {}'.format(sample_failure_cnt))
                continue
            # print(episode_generator.support_kp_categories)

            if using_crop:
                preprocess = mytransforms.Compose([
                    mytransforms.RandomCrop(crop_bbox=False),
                    mytransforms.Resize(longer_length=self.opts['square_image_length']),  # 368
                    mytransforms.CenterPad(target_size=self.opts['square_image_length']),
                    mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
                ])
            else:
                preprocess = mytransforms.Compose([
                    mytransforms.Resize(longer_length=self.opts['square_image_length']),  # 368
                    mytransforms.CenterPad(target_size=self.opts['square_image_length']),
                    mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
                ])
            image_transform = transforms.Compose([
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # transforms.RandomGrayscale(p=0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # define a list containing the paths, each path is represented by kp index pair [index1, index2]
            # our paths are subject to support keypoints; then we will interpolated kps for each path. The paths is possible to be empty list []
            num_random_paths = self.opts['num_random_paths']  # only used when auxiliary_path_mode='random'
            # path_mode:  'exhaust', 'predefined', 'random'
            auxiliary_paths = get_auxiliary_paths(path_mode=self.opts['auxiliary_path_mode'], support_keypoint_categories=episode_generator.support_kp_categories, num_random_paths=num_random_paths)

            support_dataset = AnimalPoseDataset(episode_generator.supports,
                                                episode_generator.support_kp_categories,
                                                using_auxiliary_keypoints=False,  # self.opts['use_interpolated_kps']
                                                interpolation_knots=self.opts['interpolation_knots'],
                                                interpolation_mode=self.opts['interpolation_mode'],
                                                auxiliary_path=auxiliary_paths,
                                                hdf5_images_path=self.opts['hdf5_images_path'],
                                                saliency_maps_root=self.opts['saliency_maps_root'],
                                                output_saliency_map=False,  # self.opts['use_pum']
                                                preprocess=preprocess,
                                                input_transform=image_transform
                                                )

            query_dataset = AnimalPoseDataset(episode_generator.queries,
                                              episode_generator.support_kp_categories,
                                              using_auxiliary_keypoints=False,  # self.opts['use_interpolated_kps']
                                              interpolation_knots=self.opts['interpolation_knots'],
                                              interpolation_mode=self.opts['interpolation_mode'],
                                              auxiliary_path=auxiliary_paths,
                                              hdf5_images_path=self.opts['hdf5_images_path'],
                                              saliency_maps_root=self.opts['saliency_maps_root'],
                                              output_saliency_map=False,  # self.opts['use_pum']
                                              preprocess=preprocess,
                                              input_transform=image_transform
                                              )

            support_loader = DataLoader(support_dataset, batch_size=self.opts['K_shot'], shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.opts['M_query'], shuffle=False)

            support_loader_iter = iter(support_loader)
            query_loader_iter = iter(query_loader)
            (supports, support_labels, support_kp_mask, support_scale_trans, _, _, support_saliency, _, _) = support_loader_iter.next()
            (queries, query_labels, query_kp_mask, query_scale_trans, _, _, query_saliency, query_bbx_origin, query_w_h_origin) = query_loader_iter.next()

            # make_grid_images(supports, denormalize=True, save_path='grid_image_s.jpg')
            # make_grid_images(queries, denormalize=True, save_path='grid_image_q.jpg')
            # make_grid_images(support_saliency.cuda(), denormalize=False, save_path='./ss.jpg')
            # make_grid_images(query_saliency.cuda(), denormalize=False, save_path='./sq.jpg')
            # print(episode_generator.supports)
            # save_episode_before_preprocess(episode_generator, episode_i, delete_old_files=False, draw_main_kps=True, draw_interpolated_kps=False, interpolation_knots=self.opts['interpolation_knots'], interpolation_mode=self.opts['interpolation_mode'], path_mode='predefined')
            # show_save_episode(supports, support_labels, support_kp_mask, queries, query_labels, query_kp_mask, episode_generator, episode_i,
            #                   is_show=False, is_save=True, delete_old_files=True)
            # support_kp_mask = episode_generator.support_kp_mask  # B1 x N
            # query_kp_mask = episode_generator.query_kp_mask  # B2 x N
            if torch.cuda.is_available():
                supports, queries = supports.cuda(), queries.cuda()
                support_labels, query_labels = support_labels.float().cuda(), query_labels.cuda()  # .float().cuda()
                support_kp_mask = support_kp_mask.cuda()  # B1 x N
                query_kp_mask = query_kp_mask.cuda()  # B2 x N

            if self.opts['use_body_part_protos'] and self.opts['eval_with_body_part_protos']:
                # retrieve index, which makes the retrieved body part order fit current oder, in case "order_fixed = False", namely the dynamic support kp categories
                order_index = [KEYPOINT_TYPES.index(kp_type) for kp_type in episode_generator.support_kp_categories]
                proto_mask_temp = copy.deepcopy(proto_mask)    # N
                proto_mask_temp = proto_mask_temp[order_index] # N, retrieve from standard order to fit current order in case dynamic support_kp_categories
                B1 = supports.shape[0]
                support_kp_mask = proto_mask_temp.repeat(B1, 1)  # B1 x N, overwrite support_kp_mask, union_support_kp_mask

            # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
            union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0  # N
            # compute the valid query keypoints, using broadcast
            valid_kp_mask = (query_kp_mask * union_support_kp_mask.reshape(1, -1))  # B2 x N
            num_valid_kps = torch.sum(valid_kp_mask)
            num_valid_kps_for_samples = torch.sum(valid_kp_mask, dim=1)  # B2
            gp_valid_samples_mask = num_valid_kps_for_samples >= episode_generator.least_query_kps_num  # B2
            gp_num_valid_kps = torch.sum(num_valid_kps_for_samples * gp_valid_samples_mask)
            gp_valid_kp_mask = valid_kp_mask * gp_valid_samples_mask.reshape(-1, 1)  # (B2 x N) * (B2, 1) = (B2 x N)

            # check #1    there may exist some wrongly labeled images where the keypoints are outside the boundary
            if torch.any(support_labels > 1) or torch.any(support_labels < -1) or torch.any(query_labels > 1) or torch.any(query_labels < -1):
                continue             # skip current episode directly
            # check #2
            if num_valid_kps ==  0:  # flip transform may lead zero intersecting keypoints between support and query, namely zero valid kps
                continue             # skip directly

            if finetuning_steps > 0:
                self.finetuning_via_gradient_steps(finetuning_steps, original_params_dict, supports, support_labels, support_kp_mask)


            if self.encoder_type == 0:  # Encoder, EncoderMultiScaleType1
                # feature and semantic distinctiveness, note that x2 = support_saliency.cuda() or query_saliency.cuda()
                support_features, support_lateral_out = self.encoder(x=supports, x2=None, enable_lateral_output=False)   # B1 x C x H x W, B1 x 1 x H x W
                query_features, query_lateral_out = self.encoder(x=queries, x2=None, enable_lateral_output=False)   # B2 x C x H x W, B2 x 1 x H x W

                # attentive_features, attention_maps_l2, attention_maps_l1, _ = \
                #     feature_modulator2(support_features, support_labels, support_kp_mask, query_features, context_mode=self.context_mode,
                #                        sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'],
                #                        fused_attention=self.opts['use_fused_attention'], output_attention_maps=False)


                if self.opts['use_body_part_protos'] and self.opts['eval_with_body_part_protos']:
                    # we should use body_part_proto and proto_mask (updated at above)
                    avg_support_repres = copy.deepcopy(body_part_proto)    # C x N
                    avg_support_repres = avg_support_repres[:, order_index]  # N, retrieve from standard order to fit current order in case dynamic support_kp_categories
                else:
                    # B1 x C x N
                    support_repres, conv_query_features = extract_representations(support_features, support_labels, support_kp_mask, context_mode=self.context_mode,
                                    sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'], together_trans_features=None)
                    avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N

                attentive_features, attention_maps_l2, attention_maps_l1, _ = feature_modulator3(avg_support_repres, query_features, \
                                       fused_attention=self.opts['use_fused_attention'], output_attention_maps=True, compute_similarity=False)

                # show_save_attention_maps(attention_maps_l2, queries, support_kp_mask, query_kp_mask, episode_generator.support_kp_categories,
                #                          episode_num=episode_i, is_show=False, is_save=True, delete_old=False, T=query_scale_trans)
                # show_save_attention_maps(attention_maps_l1, queries, support_kp_mask, query_kp_mask, episode_generator.support_kp_categories,
                #                          episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/attention_maps2', delete_old=False, T=query_scale_trans)

                # p_support_lateral_out = self.numerical_transformation(support_lateral_out, w_computing_mode=2)  # transform into positive number, w^-1
                # show_save_distinctive_maps(p_support_lateral_out, supports, heatmap_process_method=2, episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/distinctive_maps_s', delete_old=False)
                # p_query_lateral_out = self.numerical_transformation(query_lateral_out, w_computing_mode=2)  # transform into positive number, w^-1
                # show_save_distinctive_maps(p_query_lateral_out, queries, heatmap_process_method=2, episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/distinctive_maps_q', delete_old=False)
                # inv_w_patches, w_patches = self.get_distinctiveness_for_parts(p_support_lateral_out, p_query_lateral_out, support_labels.float(), support_kp_mask, query_labels.float())
                # inv_w_patches = inv_w_patches.cpu().detach().numpy()


            elif self.encoder_type == 1:  # can only converge when using feature_modulator2()
                attentive_features = self.encoder(supports, queries, support_labels, support_kp_mask)

            keypoint_descriptors = self.descriptor_net(attentive_features)  # B2 x N x D or B2 x N x c x h x w
            # print(keypoint_descriptors.shape)

            if self.regression_type == 'direct_regression_gridMSE':
                B2 = query_kp_mask.shape[0]  # self.opts['M_query']
                N = query_kp_mask.shape[1]  # self.opts['N_way']

                mean_predictions = 0
                # mean_predictions_grid = 0  # only used for testing grid classification
                scale_num = len(self.grid_length)
                mean_grid_var = 0
                multi_scale_covar = []
                multi_scale_predictions = []
                for scale_i, grid_length in enumerate(self.grid_length):
                    if self.opts['eval_compute_var'] == 0:  # don't compute var or covar
                        predict_grids, predict_deviations, _, _= self.regressor[scale_i](keypoint_descriptors, training_phase=False)  # B2 x N x (grid_length ** 2), B2 x N x 2
                    else:  # True
                        predict_grids, predict_deviations, rho, grid_var= self.regressor[scale_i](keypoint_descriptors, training_phase=False, compute_grid_var=True)  # B2 x N x (grid_length ** 2), B2 x N x 2



                    # compute predicted keypoint locations using grids and deviations
                    predict_grids = torch.max(predict_grids, dim=2)[1]  # B2 x N, 0 ~ grid_length * grid_length - 1
                    predict_gridxy = torch.FloatTensor(B2, N, 2).cuda()
                    predict_gridxy[:, :, 0] = predict_grids % grid_length  # grid x
                    predict_gridxy[:, :, 1] = predict_grids // grid_length # grid y

                    # Method 1, deviation range: -1~1
                    predictions = (((predict_gridxy + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
                    # Method 2, deviation range: 0~1
                    # predictions = ((predict_gridxy + predict_deviations) / grid_length - 0.5) * 2
                    mean_predictions += predictions
                    # mean_predictions_grid += ((predict_gridxy + 0.5) / grid_length - 0.5) * 2  # only used for testing grid classification
                    multi_scale_predictions.append(predictions)
                    if self.opts['eval_compute_var'] == 1:  # compute uncorrelated var, B x N x 2
                        # compute variance for regression
                        # reg_var   = torch.exp(rho) / torch.tensor(np.pi*2)  # sigma^2, variance for Gaussian, rho = log(2PI*vairance)
                        reg_var = (torch.exp(rho.cpu().detach()) ** 2) / 2.0  # 2b^2, variance for laplacian, rho = log(2b)
                        # mean_grid_var += ((square_image_length / grid_length)**2) * grid_var
                        mean_grid_var += ((square_image_length / grid_length) ** 2) * (grid_var+ reg_var / 4)  # deviation regression range -1~1, therefore divided by 2^2=4 to transform to 0~1
                    elif self.opts['eval_compute_var'] == 2:  # compute covar, B x N x (2d); note for covar's case grid_var == None
                        d = rho.shape[2] // 2  # covariance latent Q: 2 x d
                        Q = rho.cpu().detach().reshape(B2, N, 2, d)
                        reg_var = torch.zeros(B2, N, 2, 2, requires_grad=False)
                        for i in range(B2):
                            for j in range(N):
                                q = Q[i, j]
                                omega = q.matmul(q.permute(1, 0)) / d  # precision matrix inv(Covar)
                                reg_var[i, j] = torch.inverse(omega)       # 2 x 2, Covar
                        covar_temp = ((square_image_length / grid_length) ** 2) * (reg_var / 4)
                        mean_grid_var += covar_temp
                        multi_scale_covar.append(covar_temp)

                mean_predictions /= scale_num
                # mean_predictions_grid /= scale_num  # only used for testing grid classification
                if self.opts['eval_compute_var'] != 0:
                    mean_grid_var   /= scale_num


                # predictions2 = predictions * (valid_kp_mask.unsqueeze(dim=2))
                predictions = mean_predictions * (valid_kp_mask.unsqueeze(dim=2))
                query_labels = query_labels * (valid_kp_mask.unsqueeze(dim=2))
                # predictions_grid = mean_predictions_grid * (valid_kp_mask.unsqueeze(dim=2))  # only used for testing grid classification

                # predictions = predictions2
                # query_labels = query_labels2

            predictions = predictions.cpu().detach()
            query_labels = query_labels.cpu().detach()

            # version = 'preprocessed', 'original'
            # show_save_predictions(queries, query_labels, valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=None,
            #                       version='original', is_show=False, is_save=True, folder_name='query_gt', save_file_prefix='eps{}'.format(episode_i))
            # show_save_predictions(queries, predictions, valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=mean_grid_var, confidence_scale=3,
            #                       version='original', is_show=False, is_save=True, folder_name='query_predict', save_file_prefix='eps{}'.format(episode_i), kp_labels_gt=query_labels)
            # # multi-scale covar & predictions
            # for scale_i, grid_length in enumerate(self.grid_length):
            #     show_save_predictions(queries, multi_scale_predictions[scale_i].cpu().detach(), valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=multi_scale_covar[scale_i], confidence_scale=3,
            #                       version='original', is_show=False, is_save=True, folder_name='query_predict_scale%d'%(grid_length), save_file_prefix='eps{}'.format(episode_i), kp_labels_gt=query_labels)

            # show_save_warp_images(queries, query_labels, predictions, valid_kp_mask, query_scale_trans, support_labels.cpu().detach(), support_scale_trans, episode_generator,
            #                       square_image_length, version='original', is_show=False, is_save=True, save_file_prefix=episode_i)  # 'original' or 'preprocessed'
            # show_save_warp_images(queries, query_labels, predictions, valid_kp_mask, query_scale_trans, support_labels.cpu().detach(), support_scale_trans, episode_generator,
            #                       square_image_length, kp_var=mean_grid_var, confidence_scale=3, version='original', is_show=False, is_save=True, save_file_prefix=episode_i)  # 'original' or 'preprocessed'

            B2 = query_kp_mask.shape[0]  # self.opts['M_query']
            N = query_kp_mask.shape[1]  # self.opts['N_way']

            # square distance diff in original image scale
            predictions_o = recover_kps(predictions, square_image_length, query_scale_trans)
            query_labels_o = recover_kps(query_labels, square_image_length, query_scale_trans)
            square_diff = torch.sum((predictions_o - query_labels_o) ** 2, dim=2).cpu().detach().numpy()  # B2 x M
            # square_diff2 = torch.sum((predictions / 2 - query_labels /2) ** 2, dim=2).cpu().detach().numpy()  # B2 x M
            if pck_thresh_type == 'bbx':
                longer_edge = np.max(query_bbx_origin[:, [2, 3]].numpy(), axis=1)  # B2, query_bbx_origin's format xmin, ymin, w, h
            else:  # == 'img'
                longer_edge = np.max(query_w_h_origin.numpy(), axis=1)
            longer_edge = longer_edge.reshape(-1, 1)  # B2 x 1

            # square_diff2 = torch.sum((predictions_grid / 2 - query_labels / 2) ** 2, 2).cpu().detach().numpy()
            # correct += np.sum(square_diff < sqaure_error_tolerance)
            # total += query_labels.shape[0] * query_labels.shape[1] # equal to opts['M_query'] * opts['N_way']
            result_mask = valid_kp_mask.cpu().detach().numpy().astype(np.bool)
            if eval_method == 'method1':
                for ind, thr in enumerate(pck_thresh):
                    judges = (square_diff <= (thr * longer_edge)**2)
                    # judges = (square_diff2 < (thr ) ** 2)

                    judges = judges.reshape(-1)
                    # masking
                    judges = judges[result_mask.reshape(-1)]
                    tps[ind].extend(judges)
                    fps[ind].extend(1 - judges)

                    acc_cur = np.sum(judges) / len(judges)
                    acc_list[ind].append(acc_cur)

                if (episode_i % 20 == 0 or episode_i == (num_test_episodes - 1)) and len(tps[0]) > 0:
                    # recall, AP = compute_recall_ap(tps, fps, len(tps[0]))
                    acc_mean, interval = mean_confidence_interval_multiple(acc_list)

            # ----------------------------------------------------------------------------
            # used to compute relationship between normalized distance error and localization uncertainty
            # np_mean_grid_var = mean_grid_var.numpy()
            # d_error = np.sqrt(square_diff) / longer_edge
            # count_mask = (d_error < d_norm_max) * result_mask  # B x N
            # for ii in range(B2):
            #     for jj in range(N):
            #         if count_mask[ii, jj] == False:
            #             continue
            #         bin_i = int(d_error[ii, jj] / d_increment)
            #         d_uc_bins[bin_i, 0] += d_error[ii, jj]
            #         uc_tr = np_mean_grid_var[ii, jj, 0, 0] + np_mean_grid_var[ii, jj, 1, 1]
            #         uc_det = np_mean_grid_var[ii, jj, 0, 0]*np_mean_grid_var[ii, jj, 1, 1] - np_mean_grid_var[ii, jj, 0, 1]*np_mean_grid_var[ii, jj, 1, 0]
            #         uc_energy = 3 * np.sqrt(uc_tr + 2*np.sqrt(uc_det)) / longer_edge[ii, 0]
            #         d_uc_bins[bin_i, 1] += uc_energy
            #         d_uc_bins[bin_i, 3] += 1
            #         d_uc_bins[bin_i, 2] += inv_w_patches[ii, jj]  # semantic uncertainty
            # ----------------------------------------------------------------------------

            if episode_i % 20 == 0 or episode_i == num_test_episodes-1:
                # sum_tps = np.sum(np.array(tps), axis=1)
                # in order to display results property, like 200/200
                if episode_i == (num_test_episodes - 1):
                    episode_i = episode_i + 1
                if self.regression_type == 'direct_regression_gridMSE':
                    print('episode {}/{}, Acc {}, Int. {}, time: {}'.format(episode_i, num_test_episodes, acc_mean, interval, datetime.datetime.now()))
                    # if eval_method == 'method1':
                    #     print('episode {}/{}, Acc {}, AP {}'.format(episode_i, num_test_episodes, recall3, AP3))
                    #     print('episode {}/{}, Acc {}, AP {}'.format(episode_i, num_test_episodes, recall4, AP4))

            # increment in episode_i
            episode_i += 1

        sum_tps = np.sum(np.array(tps), axis=1)
        recall, AP = compute_recall_ap(tps, fps, len(tps[0]))
        print('episode {}/{}, Acc2 {}, AP2 {}, {}/{}, time: {}'.format(num_test_episodes, num_test_episodes, recall, AP, sum_tps, len(tps[0]), datetime.datetime.now()))
        print('==============testing end================')
        if self.logging != None and episode_i == num_test_episodes-1:
            logging.info('==============testing start==============')
            if self.regression_type == 'direct_regression' or self.regression_type == 'direct_regression_gridMSE' or self.regression_type == 'multiple_regression' or self.regression_type == 'composite_regression':
                logging.info('episode {}/{}, Acc {}, AP {}, {}/{}, time: {}'.format(episode_i,
                    num_test_episodes, recall, AP, sum_tps, len(tps[0]), datetime.datetime.now()))
            logging.info('==============testing end===============')

        if self.opts['set_eval']:
            self.set_train_status()  # important
        torch.set_grad_enabled(True)  # enable grad computation

        # # ----------------------------------------------------------------------------
        # fig = plt.figure(111)
        # uc_data = np.copy(d_uc_bins)
        # uc_data[:, [0, 1]] /= uc_data[:, 3].reshape(-1, 1)  # compute average
        # print("bin count: ", uc_data[:, 3])
        # plt.plot(uc_data[:, 0], uc_data[:, 1], marker='.', color='black')
        # ax = fig.get_axes()
        # ax[0].set_xticks(np.linspace(0, d_norm_max, N_bins+1))
        # plt.xlim([0, d_norm_max])
        # plt.grid(True)
        # # plt.show()
        # plt.savefig('d-uc.pdf', bbox_inches='tight')
        # # np.save('d-uc.npy', d_uc_bins)
        # # ----------------------------------------------------------------------------

        return acc_mean, recall #, d_uc_bins

    # @torch.no_grad()
    def validate2(self, episode_generator, num_test_episodes=100, eval_method='method1', using_crop='True'):
        torch.set_grad_enabled(False)  # disable grad computation
        if episode_generator == None:
            return

        print('==============testing start==============')
        ks_sigmas = np.array([.25, .25, .35, .35, .26,
                              1.07, 1.07, 1.07,
                              1.07, 1.07, 1.07, 1.07,
                              .87, .87, .87, .87,
                              .89, .89, .89, .89]) / 10.0
        # specified_kp_ids = [KEYPOINT_TYPE_IDS[kp_type] for kp_type in episode_generator.support_kp_categories]
        # specified_ks_sigmas = ks_sigmas[specified_kp_ids]

        square_image_length = self.opts['square_image_length']  # 368
        pck_thresh_bbx = np.array([0.1, 0.15])
        # pck_thresh_img = np.array([20.0 / 368, 40.0 / 368])  # 0.06 * 384 = 23.04 pixels (23 pixels)
        pck_thresh_img = np.array([0.06, 0.1])  # 0.06 * 384 = 23.04 pixels (23 pixels)
        pck_thresh_type = 'bbx'  # 'bbx' or 'img'
        if pck_thresh_type == 'bbx':  # == 'bbx'
            pck_thresh = pck_thresh_bbx
        else:  # == 'img'
            pck_thresh = pck_thresh_img

        ks_thresh = np.array([0.5, 0.25])
        tps, fps = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]
        tps2, fps2 = [[] for _ in range(len(ks_thresh))], [[] for _ in range(len(ks_thresh))]
        acc_list = [[] for _ in range(len(pck_thresh))]

        test_for_interpolated_kps = True
        if test_for_interpolated_kps:
            aux_tps, aux_fps = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]
            # aux_tps2, aux_fps2 = [[] for _ in range(len(ks_thresh))], [[] for _ in range(len(ks_thresh))]
            aux_acc_list = [[] for _ in range(len(pck_thresh))]

        # tps3, fps3 = [[] for _ in range(len(square_error_tolerance))], [[] for _ in range(len(square_error_tolerance))]
        # tps4, fps4 = [[] for _ in range(len(square_error_tolerance))], [[] for _ in range(len(square_error_tolerance))]

        if self.opts['set_eval']:
            self.set_eval_status()
        episode_i = 0
        sample_failure_cnt = 0
        # ---
        finetuning_steps = self.opts['finetuning_steps']
        if finetuning_steps > 0:
            original_params_dict = self.copy_original_params()
        # ---
        if self.opts['use_body_part_protos'] and self.opts['eval_with_body_part_protos']:
            self.load_proto_memory()
            body_part_proto = self.memory['proto']  # C x N
            proto_mask = self.memory['proto_mask']  # N
            aux_body_part_proto = self.memory['aux_proto']  # C x T or C x 0 if no aux kps
            aux_proto_mask = self.memory['aux_proto_mask']  # T or 0 if no aux kps
        # ---
        while episode_i < num_test_episodes:
            # roll-out an episode
            if (False == episode_generator.episode_next()):
                sample_failure_cnt += 1
                if sample_failure_cnt % 500 == 0:
                    print('sample failure times: {}'.format(sample_failure_cnt))
                continue
            # print(episode_generator.support_kp_categories)

            if using_crop:
                preprocess = mytransforms.Compose([
                    mytransforms.RandomCrop(crop_bbox=False),
                    mytransforms.Resize(longer_length=self.opts['square_image_length']),  # 368
                    mytransforms.CenterPad(target_size=self.opts['square_image_length']),
                    mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
                ])
            else:
                preprocess = mytransforms.Compose([
                    mytransforms.Resize(longer_length=self.opts['square_image_length']),  # 368
                    mytransforms.CenterPad(target_size=self.opts['square_image_length']),
                    mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
                ])
            image_transform = transforms.Compose([
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # transforms.RandomGrayscale(p=0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # define a list containing the paths, each path is represented by kp index pair [index1, index2]
            # our paths are subject to support keypoints; then we will interpolated kps for each path. The paths is possible to be empty list []
            num_random_paths = self.opts['num_random_paths']  # only used when auxiliary_path_mode='random'
            # path_mode:  'exhaust', 'predefined', 'random'
            auxiliary_paths = get_auxiliary_paths(path_mode=self.opts['auxiliary_path_mode'], support_keypoint_categories=episode_generator.support_kp_categories, num_random_paths=num_random_paths)

            support_dataset = AnimalPoseDataset(episode_generator.supports,
                                                episode_generator.support_kp_categories,
                                                using_auxiliary_keypoints=test_for_interpolated_kps,  # self.opts['use_interpolated_kps']
                                                interpolation_knots=self.opts['interpolation_knots'],
                                                interpolation_mode=self.opts['interpolation_mode'],
                                                auxiliary_path=auxiliary_paths,
                                                hdf5_images_path=self.opts['hdf5_images_path'],
                                                saliency_maps_root=self.opts['saliency_maps_root'],
                                                output_saliency_map=False,  # self.opts['use_pum']
                                                preprocess=preprocess,
                                                input_transform=image_transform
                                                )

            query_dataset = AnimalPoseDataset(episode_generator.queries,
                                              episode_generator.support_kp_categories,
                                              using_auxiliary_keypoints=test_for_interpolated_kps,  # self.opts['use_interpolated_kps']
                                              interpolation_knots=self.opts['interpolation_knots'],
                                              interpolation_mode=self.opts['interpolation_mode'],
                                              auxiliary_path=auxiliary_paths,
                                              hdf5_images_path=self.opts['hdf5_images_path'],
                                              saliency_maps_root=self.opts['saliency_maps_root'],
                                              output_saliency_map=False,  # self.opts['use_pum']
                                              preprocess=preprocess,
                                              input_transform=image_transform
                                              )

            support_loader = DataLoader(support_dataset, batch_size=self.opts['K_shot'], shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=self.opts['M_query'], shuffle=False)

            support_loader_iter = iter(support_loader)
            query_loader_iter = iter(query_loader)

            (supports, support_labels, support_kp_mask, _, support_aux_kps, support_aux_kp_mask, support_saliency, _, _) = support_loader_iter.next()
            (queries, query_labels, query_kp_mask, query_scale_trans, query_aux_kps, query_aux_kp_mask, query_saliency, query_bbx_origin, query_w_h_origin) = query_loader_iter.next()

            # make_grid_images(supports, denormalize=True, save_path='grid_image_s.jpg')
            # make_grid_images(queries, denormalize=True, save_path='grid_image_q.jpg')
            # make_grid_images(support_saliency.cuda(), denormalize=False, save_path='./ss.jpg')
            # make_grid_images(query_saliency.cuda(), denormalize=False, save_path='./sq.jpg')
            # print(episode_generator.supports)
            # save_episode_before_preprocess(episode_generator, episode_i, delete_old_files=True, draw_main_kps=True, draw_interpolated_kps=False)
            # show_save_episode(supports, support_labels, support_kp_mask, queries, query_labels, query_kp_mask, episode_generator, episode_i,
            #                   is_show=False, is_save=True, delete_old_files=True)

            # support_kp_mask = episode_generator.support_kp_mask  # B1 x N
            # query_kp_mask = episode_generator.query_kp_mask  # B2 x N
            if torch.cuda.is_available():
                supports, queries = supports.cuda(), queries.cuda()
                support_labels, query_labels = support_labels.float().cuda(), query_labels.cuda()  # .float().cuda()
                support_kp_mask = support_kp_mask.cuda()  # B1 x N
                query_kp_mask = query_kp_mask.cuda()  # B2 x N

            if test_for_interpolated_kps:
                support_aux_kps = support_aux_kps.float().cuda()  # B1 x T x 2, T = (N_paths * N_knots), total number of auxiliary keypoints
                support_aux_kp_mask = support_aux_kp_mask.cuda()  # B1 x T
                query_aux_kps = query_aux_kps.float().cuda()      # B2 x T x 2
                query_aux_kp_mask = query_aux_kp_mask.cuda()      # B2 x T

            if self.opts['use_body_part_protos'] and self.opts['eval_with_body_part_protos']:
                # retrieve index, which makes the retrieved body part order fit current oder, in case "order_fixed = False", namely the dynamic support kp categories
                order_index = [KEYPOINT_TYPES.index(kp_type) for kp_type in episode_generator.support_kp_categories]
                proto_mask_temp = copy.deepcopy(proto_mask)    # N
                proto_mask_temp = proto_mask_temp[order_index] # N, retrieve from standard order to fit current order in case dynamic support_kp_categories
                B1 = supports.shape[0]
                support_kp_mask = proto_mask_temp.repeat(B1, 1)  # B1 x N, overwrite support_kp_mask, union_support_kp_mask
                # here we suppose the aux kp order is fixed (should set the 'order_fixed = True' in main.py)
                if test_for_interpolated_kps:
                    aux_proto_mask_temp = copy.deepcopy(aux_proto_mask)  # T
                    support_aux_kp_mask = aux_proto_mask_temp.repeat(B1, 1)  # B1 x T, overwrite support_aux_kp_mask, union_support_aux_kp_mask


            # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
            union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0  # N
            # compute the valid query keypoints, using broadcast
            valid_kp_mask = (query_kp_mask * union_support_kp_mask.reshape(1, -1))  # B2 x N
            num_valid_kps = torch.sum(valid_kp_mask)
            if test_for_interpolated_kps:
                num_valid_support_aux_kps = torch.sum(support_aux_kp_mask)  # only for support auxiliary kps
                union_support_aux_kp_mask = torch.sum(support_aux_kp_mask, dim=0) > 0  # T
                valid_aux_kp_mask = query_aux_kp_mask * union_support_aux_kp_mask.reshape(1, -1)  # B x T
                num_valid_aux_kps = torch.sum(valid_aux_kp_mask)

            num_valid_kps_for_samples = torch.sum(valid_kp_mask, dim=1)  # B2
            gp_valid_samples_mask = num_valid_kps_for_samples >= episode_generator.least_query_kps_num  # B2
            gp_num_valid_kps = torch.sum(num_valid_kps_for_samples * gp_valid_samples_mask)
            gp_valid_kp_mask = valid_kp_mask * gp_valid_samples_mask.reshape(-1, 1)  # (B2 x N) * (B2, 1) = (B2 x N)

            # check #1    there may exist some wrongly labeled images where the keypoints are outside the boundary
            if torch.any(support_labels > 1) or torch.any(support_labels < -1) or torch.any(query_labels > 1) or torch.any(query_labels < -1):
                continue             # skip current episode directly
            # check #2
            if num_valid_kps ==  0:  # flip transform may lead zero intersecting keypoints between support and query, namely zero valid kps
                continue             # skip directly

            if finetuning_steps > 0:
                self.finetuning_via_gradient_steps(finetuning_steps, original_params_dict, supports, support_labels, support_kp_mask)

            if self.encoder_type == 0:  # Encoder, EncoderMultiScaleType1
                # feature and semantic distinctiveness, note that x2 = support_saliency.cuda() or query_saliency.cuda()
                support_features, support_lateral_out = self.encoder(x=supports, x2=None, enable_lateral_output=False)   # B1 x C x H x W, B1 x 1 x H x W
                query_features, query_lateral_out = self.encoder(x=queries, x2=None, enable_lateral_output=False)   # B2 x C x H x W, B2 x 1 x H x W

                # attentive_features, attention_maps_l2, attention_maps_l1, _ = \
                #     feature_modulator2(support_features, support_labels, support_kp_mask, query_features, context_mode=self.context_mode,
                #                        sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'],
                #                        fused_attention=self.opts['use_fused_attention'], output_attention_maps=False)

                if self.opts['use_body_part_protos'] and self.opts['eval_with_body_part_protos']:
                    # we should use body_part_proto and proto_mask (updated at above)
                    avg_support_repres = copy.deepcopy(body_part_proto)    # C x N
                    avg_support_repres = avg_support_repres[:, order_index]  # N, retrieve from standard order to fit current order in case dynamic support_kp_categories
                else:
                    # B1 x C x N
                    support_repres, conv_query_features = extract_representations(support_features, support_labels, support_kp_mask, context_mode=self.context_mode,
                                    sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'], together_trans_features=None)
                    avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
                attentive_features, attention_maps_l2, attention_maps_l1, _ = feature_modulator3(avg_support_repres, query_features, \
                                       fused_attention=self.opts['use_fused_attention'], output_attention_maps=False, compute_similarity=False)

                if test_for_interpolated_kps and num_valid_support_aux_kps > 0:
                    if self.opts['use_body_part_protos'] and self.opts['eval_with_body_part_protos']:
                        avg_support_repres_aux = copy.deepcopy(aux_body_part_proto)  # C x T
                    else:
                        # B1 x C x M
                        support_repres_aux, _ = extract_representations(support_features, support_aux_kps, support_aux_kp_mask, context_mode=self.context_mode,
                                           sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'])
                        avg_support_repres_aux = average_representations2(support_repres_aux, support_aux_kp_mask)
                    attentive_features_aux, _, _, _ = feature_modulator3(avg_support_repres_aux, query_features, \
                                   fused_attention=self.opts['use_fused_attention'], output_attention_maps=False, compute_similarity=False)

                # show_save_attention_maps(attention_maps_l2, queries, support_kp_mask, query_kp_mask, episode_generator.support_kp_categories,
                #                          episode_num=episode_i, is_show=False, is_save=True, delete_old=False, T=query_scale_trans)
                # show_save_attention_maps(attention_maps_l1, queries, support_kp_mask, query_kp_mask, episode_generator.support_kp_categories,
                #                          episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/attention_maps2', delete_old=False, T=query_scale_trans)

                # p_support_lateral_out = self.numerical_transformation(support_lateral_out, w_computing_mode=2)  # transform into positive number, w^-1
                # show_save_distinctive_maps(p_support_lateral_out, supports, heatmap_process_method=2, episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/distinctive_maps_s', delete_old=False)
                # p_query_lateral_out = self.numerical_transformation(query_lateral_out, w_computing_mode=2)  # transform into positive number, w^-1
                # show_save_distinctive_maps(p_query_lateral_out, queries, heatmap_process_method=2, episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/distinctive_maps_q', delete_old=False)

            elif self.encoder_type == 1:  # can only converge when using feature_modulator2()
                attentive_features = self.encoder(supports, queries, support_labels, support_kp_mask)

            keypoint_descriptors = self.descriptor_net(attentive_features)  # B2 x N x D or B2 x N x c x h x w
            if test_for_interpolated_kps and num_valid_support_aux_kps > 0:
                keypoint_descriptors_aux = self.descriptor_net(attentive_features_aux)  # B2 x T x D or B2 x T x c x h x w

            if self.regression_type == 'direct_regression':
                B2 = query_kp_mask.shape[0]  # self.opts['M_query']
                N =  query_kp_mask.shape[1]  # self.opts['N_way']

                predictions = self.regressor(keypoint_descriptors)  # B2 x N x 2

                # Method 2, masking, using broadcast
                # ignore_mask = (~union_support_kp_mask.reshape(1, N).repeat(B2, 1)) + valid_kp_mask  # B2 x N
                # predictions2 = predictions * ignore_mask.unsqueeze(dim=2)
                predictions2 = predictions * (valid_kp_mask.unsqueeze(dim=2))
                query_labels2 = query_labels * (valid_kp_mask.unsqueeze(dim=2))

                predictions = predictions2
                query_labels = query_labels2

            elif self.regression_type == 'direct_regression_gridMSE':
                B2 = query_kp_mask.shape[0]  # self.opts['M_query']
                N = query_kp_mask.shape[1]  # self.opts['N_way']
                if test_for_interpolated_kps and num_valid_support_aux_kps > 0:
                    T = support_aux_kp_mask.shape[1]

                mean_predictions = 0
                # mean_predictions_grid = 0  # only used for testing grid classification
                mean_predictions_aux = 0  # test for auxiliary keypoints
                scale_num = len(self.grid_length)
                mean_grid_var = 0
                for scale_i, grid_length in enumerate(self.grid_length):
                    if self.opts['eval_compute_var'] == 0:  # don't compute var or covar
                        predict_grids, predict_deviations, _, _= self.regressor[scale_i](keypoint_descriptors, training_phase=False)  # B2 x N x (grid_length ** 2), B2 x N x 2
                    else:  # True
                        predict_grids, predict_deviations, rho, grid_var= self.regressor[scale_i](keypoint_descriptors, training_phase=False, compute_grid_var=True)  # B2 x N x (grid_length ** 2), B2 x N x 2

                    # compute predicted keypoint locations using grids and deviations
                    predict_grids = torch.max(predict_grids, dim=2)[1]  # B2 x N, 0 ~ grid_length * grid_length - 1
                    predict_gridxy = torch.FloatTensor(B2, N, 2).cuda()
                    predict_gridxy[:, :, 0] = predict_grids % grid_length  # grid x
                    predict_gridxy[:, :, 1] = predict_grids // grid_length # grid y



                    # Method 1, deviation range: -1~1
                    predictions = (((predict_gridxy + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
                    # Method 2, deviation range: 0~1
                    # predictions = ((predict_gridxy + predict_deviations) / grid_length - 0.5) * 2
                    mean_predictions += predictions
                    # mean_predictions_grid += ((predict_gridxy + 0.5) / grid_length - 0.5) * 2  # only used for testing grid classification
                    if self.opts['eval_compute_var'] == 1:  # compute uncorrelated var, B x N x 2
                        # compute variance for regression
                        # reg_var   = torch.exp(rho) / torch.tensor(np.pi*2)  # sigma^2, variance for Gaussian, rho = log(2PI*vairance)
                        reg_var = (torch.exp(rho.cpu().detach()) ** 2) / 2.0  # 2b^2, variance for laplacian, rho = log(2b)
                        # mean_grid_var += ((square_image_length / grid_length)**2) * grid_var
                        mean_grid_var += ((square_image_length / grid_length) ** 2) * (grid_var+ reg_var / 4)  # deviation regression range -1~1, therefore divided by 2^2=4 to transform to 0~1
                    elif self.opts['eval_compute_var'] == 2:  # compute covar, B x N x (2d); note for covar's case grid_var == None
                        d = rho.shape[2] // 2  # covariance latent Q: 2 x d
                        Q = rho.cpu().detach().reshape(B2, N, 2, d)
                        reg_var = torch.zeros(B2, N, 2, 2, requires_grad=False)
                        for i in range(B2):
                            for j in range(N):
                                q = Q[i, j]
                                omega = q.matmul(q.permute(1, 0)) / d  # precision matrix inv(Covar)
                                reg_var[i, j] = torch.inverse(omega)       # 2 x 2, Covar
                        mean_grid_var += ((square_image_length / grid_length) ** 2) * (reg_var / 4)

                    # ---------------auxiliary keypoints-------------------
                    if test_for_interpolated_kps and num_valid_support_aux_kps > 0:
                        predict_grids_aux, predict_deviations_aux, _, _= self.regressor[scale_i](keypoint_descriptors_aux, training_phase=False)  # B2 x T x (grid_length ** 2), B2 x T x 2
                        predict_grids_aux = torch.max(predict_grids_aux, dim=2)[1]  # B2 x T, 0 ~ grid_length * grid_length - 1

                        predict_gridxy_aux = torch.FloatTensor(B2, T, 2).cuda()
                        predict_gridxy_aux[:, :, 0] = predict_grids_aux % grid_length   # grid x
                        predict_gridxy_aux[:, :, 1] = predict_grids_aux // grid_length  # grid y

                        # Method 1, deviation range: -1~1
                        predictions_aux = (((predict_gridxy_aux + 0.5) + predict_deviations_aux / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
                        # Method 2, deviation range: 0~1
                        # predictions_aux = ((predict_gridxy_aux + predict_deviations_aux) / grid_length - 0.5) * 2
                        mean_predictions_aux += predictions_aux

                mean_predictions /= scale_num
                # mean_predictions_grid /= scale_num  # only used for testing grid classification
                if self.opts['eval_compute_var'] != 0:
                    mean_grid_var   /= scale_num

                # predictions = predict_gridxy
                # # compute grid groundtruth and deviation
                # gridxy = (query_labels /2 + 0.5) * self.grid_length  # coordinate -1~1 --> 0~self.grid_length, B2 x N x 2
                # gridxy_quantized = gridxy.long().clamp(0, self.grid_length - 1)  # B2 x N x 2
                # query_labels = gridxy_quantized  # using the quantized gridxy as query labels


                # predictions2 = predictions * (valid_kp_mask.unsqueeze(dim=2))
                predictions = mean_predictions * (valid_kp_mask.unsqueeze(dim=2))
                query_labels = query_labels * (valid_kp_mask.unsqueeze(dim=2))
                # predictions_grid = mean_predictions_grid * (valid_kp_mask.unsqueeze(dim=2))  # only used for testing grid classification


                # predictions = predictions2
                # query_labels = query_labels2

                # ---------------auxiliary keypoints-------------------
                if test_for_interpolated_kps and num_valid_support_aux_kps > 0:
                    mean_predictions_aux /= scale_num
                    predictions_aux = mean_predictions_aux * (valid_aux_kp_mask.unsqueeze(dim=2))
                    query_aux_kps = query_aux_kps * (valid_aux_kp_mask.unsqueeze(dim=2))

            predictions = predictions.cpu().detach()
            # predictions = gp_predictions.cpu().detach()
            # predictions_grid = predictions_grid.cpu().detach()  # only used for testing grid classification
            query_labels = query_labels.cpu().detach()
            if test_for_interpolated_kps and num_valid_support_aux_kps > 0:
                predictions_aux, query_aux_kps = predictions_aux.cpu(), query_aux_kps.cpu()

            # version = 'preprocessed', 'original'
            # show_save_predictions(queries, query_labels, valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=None,
            #                       version='original', is_show=False, is_save=True, folder_name='query_gt', save_file_prefix='eps{}'.format(episode_i))
            # show_save_predictions(queries, predictions, valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=mean_grid_var, confidence_scale=3,
            #                       version='original', is_show=False, is_save=True, folder_name='query_predict', save_file_prefix='eps{}'.format(episode_i))
            # show_save_predictions(queries, query_aux_kps, valid_aux_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=None,
            #                       version='original', is_show=False, is_save=True, folder_name='query_aux_gt', save_file_prefix='eps{}'.format(episode_i))
            # show_save_predictions(queries, predictions_aux, valid_aux_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=None,
            #                       version='original', is_show=False, is_save=True, folder_name='query_aux_predict', save_file_prefix='eps{}'.format(episode_i))

            B2 = query_kp_mask.shape[0]  # self.opts['M_query']
            N = query_kp_mask.shape[1]  # self.opts['N_way']

            # square distance diff in original image scale
            predictions_o = recover_kps(predictions, square_image_length, query_scale_trans)
            query_labels_o = recover_kps(query_labels, square_image_length, query_scale_trans)
            square_diff = torch.sum((predictions_o - query_labels_o) ** 2, dim=2).cpu().detach().numpy()  # B2 x M
            # square_diff2 = torch.sum((predictions / 2 - query_labels /2) ** 2, dim=2).cpu().detach().numpy()  # B2 x M
            if pck_thresh_type == 'bbx':
                longer_edge = np.max(query_bbx_origin[:, [2, 3]].numpy(), axis=1)  # B2, query_bbx_origin's format xmin, ymin, w, h
            else:  # == 'img'
                longer_edge = np.max(query_w_h_origin.numpy(), axis=1)
            longer_edge = longer_edge.reshape(-1, 1)  # B2 x 1

            # square_diff2 = torch.sum((predictions_grid / 2 - query_labels / 2) ** 2, 2).cpu().detach().numpy()
            # correct += np.sum(square_diff < sqaure_error_tolerance)
            # total += query_labels.shape[0] * query_labels.shape[1] # equal to opts['M_query'] * opts['N_way']
            result_mask = valid_kp_mask.cpu().detach().numpy().astype(np.bool)
            # ---------------auxiliary keypoints-------------------
            if test_for_interpolated_kps and num_valid_support_aux_kps > 0:
                predictions_aux_o = recover_kps(predictions_aux, square_image_length, query_scale_trans)
                query_aux_kps_o = recover_kps(query_aux_kps, square_image_length, query_scale_trans)
                square_diff_aux = torch.sum((predictions_aux_o - query_aux_kps_o) ** 2, 2).cpu().detach().numpy()  # B x T
                result_mask_aux = valid_aux_kp_mask.cpu().detach().numpy().astype(np.bool)
            if eval_method == 'method1':  # pck
                for ind, thr in enumerate(pck_thresh):
                    judges = (square_diff <= (thr * longer_edge) ** 2)
                    # judges = (square_diff2 < (thr ) ** 2)

                    # judges3 = (judges[:, 0:2]).reshape(-1)
                    # result_mask3 = result_mask[:, 0:2]
                    # judges3 = judges3[result_mask3.reshape(-1)]
                    # tps3[ind].extend(judges3)
                    # fps3[ind].extend(1 - judges3)
                    # judges4 = (judges[:, 2:]).reshape(-1)
                    # result_mask4 = result_mask[:, 2:]
                    # judges4 = judges4[result_mask4.reshape(-1)]
                    # tps4[ind].extend(judges4)
                    # fps4[ind].extend(1-judges4)

                    judges = judges.reshape(-1)
                    # masking
                    judges = judges[result_mask.reshape(-1)]
                    tps[ind].extend(judges)
                    fps[ind].extend(1 - judges)

                    # judges2 = (square_diff2 < thr).reshape(-1)  # only used for testing grid classification
                    # judges2 = judges2[result_mask.reshape(-1)]
                    # tps3[ind].extend(judges2)
                    # fps3[ind].extend(1-judges2)

                    acc_cur = np.sum(judges) / len(judges)
                    acc_list[ind].append(acc_cur)

                if (episode_i % 20 == 0 or episode_i == (num_test_episodes-1)) and len(tps[0]) > 0:
                    # recall, AP = compute_recall_ap(tps, fps, len(tps[0]))
                    acc_mean, interval = mean_confidence_interval_multiple(acc_list)

                    # recall3, AP3 = compute_recall_ap(tps3, fps3, len(tps3[0]))
                    # recall4, AP4 = compute_recall_ap(tps4, fps4, len(tps4[0]))

                if test_for_interpolated_kps:
                    if num_valid_aux_kps > 0:
                        for ind, thr in enumerate(pck_thresh):
                            judges_aux = (square_diff_aux < (thr * longer_edge) ** 2)
                            judges_aux = judges_aux.reshape(-1)
                            # masking
                            judges_aux = judges_aux[result_mask_aux.reshape(-1)]
                            aux_tps[ind].extend(judges_aux)
                            aux_fps[ind].extend(1 - judges_aux)

                            acc_cur = np.sum(judges_aux) / len(judges_aux)
                            aux_acc_list[ind].append(acc_cur)

                    if (episode_i % 20 == 0 or episode_i == (num_test_episodes - 1)) and len(aux_tps[0]) > 0:
                        # recall_aux, AP_aux = compute_recall_ap(aux_tps, aux_fps, len(aux_tps[0]))
                        acc_mean_aux, interval_aux = mean_confidence_interval_multiple(aux_acc_list)

            if eval_method == 'method2':
                # eval method 2, using keypoint similarity based AP
                ks = np.zeros([B2, N])
                # retrieve ks_sigmas based on the dynamic support_kp_categories (which can be allowed to change in each iteration)
                specified_kp_ids = [KEYPOINT_TYPE_IDS[kp_type] for kp_type in episode_generator.support_kp_categories]
                specified_ks_sigmas = ks_sigmas[specified_kp_ids]
                k_var = (specified_ks_sigmas * 2) ** 2  # N
                bbx_areas = (query_bbx_origin[:, 2] * query_bbx_origin[:, 3]).numpy()  # B2, query_bbx_origin's format xmin, ymin, w, h
                bbx_areas = bbx_areas + np.spacing(1)
                e = square_diff / (2.0 * bbx_areas.reshape(-1, 1) * k_var.reshape(1, -1))
                ks = np.exp(-e)  # ks = exp(- d^2/ (2 * s^2 * k^2) )

                for ind, thr in enumerate(ks_thresh):
                    judges = (ks >= thr).reshape(-1)
                    # masking
                    judges = judges[result_mask.reshape(-1)]
                    tps2[ind].extend(judges)
                    fps2[ind].extend(1 - judges)

                    acc_cur = np.sum(judges) / len(judges)
                    acc_list[ind].append(acc_cur)

                if (episode_i % 20 == 0 or episode_i == (num_test_episodes - 1)) and len(tps2[0])>0:
                    # recall2, AP2 = compute_recall_ap(tps2, fps2, len(tps2[0]))
                    acc_mean, interval = mean_confidence_interval_multiple(acc_list)

                    # ------
                    # redundancy, just for below printing and clarification that this is another eval method
                    # recall, AP = recall2, AP2
                    # tps = tps2
                    # ------

            if episode_i % 20 == 0 or episode_i == num_test_episodes-1:
                # sum_tps = np.sum(np.array(tps), axis=1)
                # in order to display results property, like 200/200
                if episode_i == (num_test_episodes - 1):
                    episode_i = episode_i + 1
                if self.regression_type == 'direct_regression'  or self.regression_type == 'direct_regression_gridMSE' or self.regression_type == 'multiple_regression' or self.regression_type=='composite_regression':
                    print('episode {}/{}, Acc {}, Int. {}, time: {}'.format(episode_i, num_test_episodes, acc_mean, interval, datetime.datetime.now()))
                    # if eval_method == 'method1':
                    #     print('episode {}/{}, accuracy {}, AP {}'.format(episode_i, num_test_episodes, recall3, AP3))
                    #     print('episode {}/{}, accuracy {}, AP {}'.format(episode_i, num_test_episodes, recall4, AP4))
                    if test_for_interpolated_kps and len(aux_tps[0]) > 0:
                        # sum_tps_aux = np.sum(np.array(aux_tps), axis=1)
                        # print('aux-kps Acc {}, AP {}, {}/{}'.format(recall_aux, AP_aux, sum_tps_aux, len(aux_tps[0])))
                        print('aux Acc {}, Int. {}'.format(acc_mean_aux, interval_aux))

            # increment in episode_i
            episode_i += 1

        if eval_method == 'method2':  # redundancy, just for below printing and clarification that this is another eval method
            tps, fps = tps2, fps2
        sum_tps = np.sum(np.array(tps), axis=1)
        recall, AP = compute_recall_ap(tps, fps, len(tps[0]))
        print('episode {}/{}, Acc2 {}, AP2 {}, {}/{}, time: {}'.format(num_test_episodes, num_test_episodes, recall, AP, sum_tps, len(tps[0]), datetime.datetime.now()))
        if test_for_interpolated_kps and len(aux_tps[0]) > 0:
            sum_tps_aux = np.sum(np.array(aux_tps), axis=1)
            recall_aux, AP_aux = compute_recall_ap(aux_tps, aux_fps, len(aux_tps[0]))
            print('aux-kps Acc2 {}, AP2 {}, {}/{}'.format(recall_aux, AP_aux, sum_tps_aux, len(aux_tps[0])))

        print('==============testing end================')
        if self.logging != None and episode_i == num_test_episodes-1:
            logging.info('==============testing start==============')
            if self.regression_type == 'direct_regression' or self.regression_type == 'direct_regression_gridMSE' or self.regression_type == 'multiple_regression' or self.regression_type == 'composite_regression':
                logging.info('episode {}/{}, Acc {}, AP {}, {}/{}, time: {}'.format(episode_i,
                    num_test_episodes, recall, AP, sum_tps, len(tps[0]), datetime.datetime.now()))
            logging.info('==============testing end===============')

        if self.opts['set_eval']:
            self.set_train_status()  # important
        torch.set_grad_enabled(True) # enable grad computation

        if  test_for_interpolated_kps:
            if len(aux_tps[0]) > 0:
                return recall, AP, recall_aux, AP_aux
            else:
                return recall, AP, [0, 0], [0, 0]

        return acc_mean, recall

    def copy_original_params(self):
        # save a copy of the original model parameters; it should note that default keep_vars is false, which will cause require_grads=False.
        original_params_dict = {'E': copy.deepcopy(self.encoder.state_dict(keep_vars=True)), 'D': copy.deepcopy(self.descriptor_net.state_dict(keep_vars=True)),
                                'R': copy.deepcopy(self.regressor.state_dict(keep_vars=True)), 'MCovar': copy.deepcopy(self.covar_branch.state_dict(keep_vars=True))}
        return original_params_dict

    def finetuning_via_gradient_steps(self, finetuning_steps, original_params_dict, supports, support_labels, support_kp_mask):
        torch.set_grad_enabled(True)  # enable grad computation
        current_params_dict = copy.deepcopy(original_params_dict)
        # print(current_params_dict)
        # assign the current_params_dict to model
        self.encoder.load_state_dict(current_params_dict['E'])
        self.descriptor_net.load_state_dict(current_params_dict['D'])
        self.regressor.load_state_dict(current_params_dict['R'])
        self.covar_branch.load_state_dict(current_params_dict['MCovar'])
        # self.encoder, self.descriptor_net, self.regressor, self.covar_branch = self.encoder.cuda(), self.descriptor_net.cuda(), self.regressor.cuda(), self.covar_branch.cuda()
        # e = nn.ParameterList(self.encoder.parameters())
        self.optimizer_init(lr=0.0001, lr_auxiliary = 0.0001, weight_decay=0, optimization_algorithm='Adam')
        # update_lr = 0.0001
        union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0  # N
        for k in range(finetuning_steps):
            # print_weights(self.regressor[0].linear_grid_class[0].weight.data)
            support_features, support_lateral_out = self.encoder(x=supports, x2=None, enable_lateral_output=True)   # B1 x C x H x W, B1 x 1 x H x W
            if self.opts['use_pum']:
                # computing_inv_w = True
                w_computing_mode = 2
                p_support_lateral_out = self.numerical_transformation(support_lateral_out, w_computing_mode=w_computing_mode)  # B1 x 1 x H' x W', transform into positive number
                # regarding main training kps
                inv_w_patches, w_patches = self.get_distinctiveness_for_parts(p_support_lateral_out, p_support_lateral_out, support_labels, support_kp_mask, support_labels)
            else:
                inv_w_patches = None

            # B1 x C x N
            support_repres, conv_query_features = extract_representations(support_features, support_labels, support_kp_mask, context_mode=self.context_mode,
                            sigma=self.opts['sigma'], downsize_factor=self.opts['downsize_factor'], image_length=self.opts['square_image_length'], together_trans_features=None)
            avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
            attentive_features, attention_maps_l2, attention_maps_l1, similarity_map = feature_modulator3(avg_support_repres, support_features, \
                                   fused_attention=self.opts['use_fused_attention'], output_attention_maps=False, compute_similarity=False)
            keypoint_descriptors = self.descriptor_net(attentive_features)  # B2 x N x D or B2 x N x c x h x w
            valid_kp_mask_for_support = (support_kp_mask * union_support_kp_mask.reshape(1, -1))  # B1 x N
            loss_grid_class, loss_deviation, _ = self.multiscale_regression(keypoint_descriptors, support_labels, valid_kp_mask_for_support, self.grid_length, weight=None, inv_w_patches=inv_w_patches, w_patches=None)
            loss = loss_grid_class + loss_deviation

            # params_list = nn.ParameterList()
            # model_list = [self.descriptor_net, self.regressor]
            # len_temp = 0
            # # combine the Parameters
            # for each_model in model_list:
            #     each_param_list = nn.ParameterList(each_model.parameters())
            #     params_list.extend(each_param_list)
            # # compute gradients; note that the Parameter in params_list should all be differentiable (namely requires_grad==True)
            # grad = torch.autograd.grad(loss, params_list)
            # # update weights
            # fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, params_list)))
            # # assign the adapted weights to the model
            # for i, each_model in enumerate(model_list):
            #     each_param_list = nn.ParameterList(each_model.parameters())
            #     for j in range(len(each_param_list)):  # Parameter has two
            #         each_param_list[j].data = fast_weights[len_temp + j]
            #     len_temp += len(each_param_list)

            self.optimizer_step(loss)
            # print(loss.item())
        torch.set_grad_enabled(False)  # disable grad computation


# show the images in one episode
def  show_save_episode(supports, support_labels, support_kp_mask, queries, query_labels, query_kp_mask, episode_generator, episode_num=0,
                      support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None, is_show=False, is_save=True, delete_old_files=False, draw_main_kps=True):
    '''
    show the supervised keypoints in the support images and query images, as well as optionally drawing interpolated keypoints

    if support_aux_kps is not none, the image will draw interpolated keypoints.
    '''

    # support_loader_iter = iter(support_loader)
    # query_loader_iter = iter(query_loader)

    # (supports, support_labels, support_kp_mask, _) = support_loader_iter.next()
    # (queries, query_labels, query_kp_mask, _) = query_loader_iter.next()

    import copy
    supports, support_labels, support_kp_mask = copy.deepcopy(supports.detach().cpu()), copy.deepcopy(support_labels.detach().cpu()), copy.deepcopy(support_kp_mask.detach().cpu())
    queries, query_labels, query_kp_mask = copy.deepcopy(queries.detach().cpu()), copy.deepcopy(query_labels.detach().cpu()), copy.deepcopy(query_kp_mask.detach().cpu())

    # whether draw interpolated keypoints
    draw_interpolated_kps = False if (support_aux_kps is None) else True
    if draw_interpolated_kps == True:
        support_aux_kps, support_aux_kp_mask = copy.deepcopy(support_aux_kps.detach().cpu()), copy.deepcopy(support_aux_kp_mask.detach().cpu())
        query_aux_kps, query_aux_kp_mask = copy.deepcopy(query_aux_kps.detach().cpu()), copy.deepcopy(query_aux_kp_mask.detach().cpu())

    # grid_image = torchvision.utils.make_grid(supports, nrow=2, padding=2, pad_value=1)
    # grid_image = grid_image.permute(1,2,0)
    # plt.imshow(grid_image)
    # plt.show()

    if is_save:
        save_image_root = './episode_images/preprocessed'

        if os.path.exists(save_image_root) == False:
            os.mkdir(save_image_root)
        if os.path.exists(save_image_root + "/" + 'support') == False:
            os.mkdir(save_image_root + "/" + 'support')
        if os.path.exists(save_image_root + "/" + 'query') == False:
            os.mkdir(save_image_root + "/" + 'query')

        # remove old episode images
        if delete_old_files == True:
            for each_file in os.listdir(os.path.join(save_image_root, 'support')):
                os.remove(os.path.join(save_image_root, 'support', each_file))
            for each_file in os.listdir(os.path.join(save_image_root, 'query')):
                os.remove(os.path.join(save_image_root, 'query', each_file))

    B1 = supports.shape[0]
    B2 = queries.shape[0]
    width = supports.shape[-1]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    r_pixels = 8
    for batch_i in range(B1):
        # support_image = supports.squeeze().permute(1, 2, 0)
        # query_image = queries.squeeze().permute(1, 2, 0)
        single_support_image, single_support_label = supports[batch_i, :, :, :], support_labels[batch_i, :, :]
        single_support_image = single_support_image.permute(1, 2, 0)

        # single_support_label = (single_support_label * (width-1)).long()
        single_support_label = ((single_support_label / 2 + 0.5) * (width - 1)).long()
        keypoints = {kp_type: single_support_label[i, :] for i, kp_type in enumerate(episode_generator.support_kp_categories) \
                     if support_kp_mask[batch_i, i] != 0 }
        print(episode_generator.supports[batch_i])
        print(keypoints)

        # support_image ranges 0~1
        for c in range(3):
            single_support_image[:, :, c] = (single_support_image[:, :, c] * std[c] + mean[c])
        im_uint8 = single_support_image.mul(255).numpy().astype(np.uint8)
        im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
        im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr
        # cv2.circle(im_uint8_bgr, (160, 169), 15, [255,255,255])
        # plt.imshow(im_uint8_bgr[:,:,::-1])
        # plt.show()

        if draw_main_kps:
            labeled_support_image = draw_skeletons(im_uint8_bgr, [keypoints], KEYPOINT_TYPES, circle_radius=r_pixels, limbs=[])
        else:
            labeled_support_image = im_uint8_bgr
        if draw_interpolated_kps == True:
            npimg_cur = np.copy(labeled_support_image)
            for j, is_visible in enumerate(support_aux_kp_mask[batch_i]):
                if is_visible == 0:
                    continue
                body_part = ((support_aux_kps[batch_i, j, :] / 2 + 0.5) * (width - 1)).long()
                center = (int(body_part[0]), int(body_part[1]))
                cv2.circle(npimg_cur, center, (int)(r_pixels/2), [0, 0, 255], thickness=-1)
                labeled_support_image = cv2.addWeighted(labeled_support_image, 0.3, npimg_cur, 0.7, 0)
        labeled_support_image = labeled_support_image[:, :, ::-1]  # bgr to rgb

        if is_show:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(single_support_image)
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(labeled_support_image)
            plt.show()

        if is_save:
            # write new episode images
            cv2.imwrite(os.path.join(save_image_root, 'support/eps{}_s_{}.jpg'.format(episode_num, batch_i)), labeled_support_image[:,:,::-1])

    for batch_i in range(B2):
        # support_image = supports.squeeze().permute(1, 2, 0)
        # query_image = queries.squeeze().permute(1, 2, 0)
        single_query_image, single_query_label = queries[batch_i, :, :, :], query_labels[batch_i, :, :]
        single_query_image = single_query_image.permute(1, 2, 0)

        # single_query_label = (single_query_label * (width-1)).long()
        single_query_label = ((single_query_label / 2 + 0.5) * (width - 1)).long()
        keypoints = {kp_type: single_query_label[i, :] for i, kp_type in
                     enumerate(episode_generator.support_kp_categories) if query_kp_mask[batch_i, i] != 0}
        print(episode_generator.queries[batch_i])
        print(keypoints)

        for c in range(3):
            single_query_image[:, :, c] = (single_query_image[:, :, c] * std[c] + mean[c])
        im_uint8 = single_query_image.mul(255).numpy().astype(np.uint8)
        im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
        im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr
        if draw_main_kps:
            labeled_query_image = draw_skeletons(im_uint8_bgr, [keypoints], KEYPOINT_TYPES, circle_radius=r_pixels, limbs=[])
        else:
            labeled_query_image = im_uint8_bgr
        if draw_interpolated_kps == True:
            npimg_cur = np.copy(labeled_query_image)
            for j, is_visible in enumerate(query_aux_kp_mask[batch_i]):
                if is_visible == 0:
                    continue
                body_part = ((query_aux_kps[batch_i, j, :] / 2 + 0.5) * (width - 1)).long()
                center = (int(body_part[0]), int(body_part[1]))
                cv2.circle(npimg_cur, center, (int)(r_pixels / 2), [0, 0, 255], thickness=-1)
                labeled_query_image = cv2.addWeighted(labeled_query_image, 0.3, npimg_cur, 0.7, 0)
        labeled_query_image = labeled_query_image[:, :, ::-1]  # bgr to rgb

        if is_show:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(single_query_image)
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(labeled_query_image)

            plt.show()

        if is_save:
            # write new episode images
            cv2.imwrite(os.path.join(save_image_root, 'query/eps{}_q_{}.jpg'.format(episode_num, batch_i)), labeled_query_image[:,:,::-1])

def show_save_predictions_arrow(images: torch.Tensor, kp_labels: torch.Tensor, kp_mask: torch.Tensor, scale_trans: torch.Tensor, episode_generator: EpisodeGenerator,
                          square_image_length=368, kp_var=None, confidence_scale=1, version='preprocessed', is_show=False, is_save=True, folder_name='query_gt', save_file_prefix="", kp_labels_gt=None, root_postfix=""):
    # queries: B2 x C x H x W
    # query_labels: B2 x N x 2
    # scale_trans: B2 x 6 (scale, x_offset, y_offset, bbx_area, pad_xoffset, pad_yoffset)
    # kp_var: B2 x N x 2 (var_x, var_y)
    # confidence_scale: confidence_scale * std (like 3*sigma)
    import copy
    images, kp_labels = copy.deepcopy(images.cpu().detach()), copy.deepcopy(kp_labels)
    kp_mask, scale_trans = copy.deepcopy(kp_mask), copy.deepcopy(scale_trans)
    draw_kp_var = True if kp_var is not None else False
    if draw_kp_var:
        # if not None, format is independent var B2 x N x 2, [var(x), var(y)]
        # or covariance B2 x N x 2 x 2, Covar = [Sigma_xx, Sigma_xy; Sigma_xy, Sigma_yy]
        kp_var = copy.deepcopy(kp_var)

    if kp_labels_gt is not None:
        kp_labels_gt = copy.deepcopy(kp_labels_gt)
        kp_labels_gt = recover_kps(kp_labels_gt, square_image_length, scale_trans)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    r_pixels = 6  # circle 8; tilted_cross: 6
    ucellipse_thickness = -1  # thickness default: 1 or 2; fill ellipse: -1
    ucellipse_alpha = 0.4    # opaque, 1 ; fill ellipse: 0.4

    save_image_root = './episode_images/predictions'+root_postfix  # root_postfix is used to distinguish diff. datasets
    if os.path.exists(save_image_root) == False:
        os.makedirs(save_image_root)
    if os.path.exists(save_image_root + "/" + folder_name) == False:
        os.makedirs(save_image_root + "/" + folder_name)

    B = kp_labels.shape[0]  # number of query image
    N = kp_labels.shape[1]  # number of keypoints
    if version == 'preprocessed':  # show keypoints in pre-processed images
        for batch_i in range(B):
            single_image, single_label = images[batch_i, :, :, :], kp_labels[batch_i, :, :]
            single_image = single_image.permute(1, 2, 0)  # H x W x C

            # single_label = (single_label * (square_image_length-1)).long()
            single_label = ((single_label / 2 + 0.5) * (square_image_length - 1)).long()
            keypoints = {kp_type: single_label[i, :] for i, kp_type in enumerate(episode_generator.support_kp_categories) \
                         if kp_mask[batch_i, i] != 0 }

            print(episode_generator.queries[batch_i])
            print(keypoints)

            # support_image ranges 0~1
            for c in range(3):
                single_image[:, :, c] = single_image[:, :, c] * std[c] + mean[c]
            im_uint8 = single_image.mul(255).numpy().astype(np.uint8)
            im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
            im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr
            # cv2.circle(im_uint8_bgr, (160, 169), 15, [255,255,255])
            # plt.imshow(im_uint8_bgr[:,:,::-1])
            # plt.show()

            # labeled_image = draw_skeletons(im_uint8_bgr, [keypoints], KEYPOINT_TYPES, marker='circle', circle_radius=r_pixels, limbs=[], alpha=0.7)
            labeled_image = draw_skeletons(im_uint8_bgr, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
            labeled_image = labeled_image[:, :, ::-1]  # bgr to rgb

            if is_show:
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(single_image)
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(labeled_image)
                plt.show()

            if is_save:
                # write labelled images
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image[:,:,::-1])
    else:  # show keypoints in original image
        import PIL.Image as Image
        # uncertainty_e
        for batch_i in range(B):
            single_label = kp_labels[batch_i, :, :]
            single_image = Image.open(episode_generator.queries[batch_i]['filename']).convert('RGB')  # H x W x C
            single_image = np.array(single_image)  # .astype(np.uint8)

            # transform the scale-padded image annotations to original annotations
            # single_label = (single_label * (square_image_length-1)).long()
            single_label = single_label / 2 + 0.5  # 0~1
            single_label *= square_image_length - 1
            single_label += scale_trans[batch_i, 1:3].repeat(N, 1)
            single_label /= scale_trans[batch_i, 0]
            single_label = single_label.long()

            if draw_kp_var:
                if len(kp_var.shape) == 3:  # independent var B2 x N x 2, [var(x), var(y)]
                    single_image_kp_var = kp_var[batch_i, :, :]
                    single_image_kp_std = torch.sqrt(single_image_kp_var)
                    single_image_kp_std /= scale_trans[batch_i, 0]  # N x 2
                    single_image_kp_angle = None
                elif len(kp_var.shape) == 4: # covariance B2 x N x 2 x 2, Covar = [Sigma_xx, Sigma_xy; Sigma_xy, Sigma_yy]
                    single_image_kp_covar = kp_var[batch_i, :, :, :]  # N x 2 x 2
                    single_image_kp_std = torch.zeros(N, 2)  # N x 2
                    single_image_kp_angle = torch.zeros(N)   # N
                    for kp_j in range(N):
                        e, _, angle = compute_eigenvalues(single_image_kp_covar[kp_j])
                        single_image_kp_std[kp_j, :] = torch.sqrt(e[:, 0])  # sqrt(major_axis, minor_axis)
                        single_image_kp_angle[kp_j] = angle

                    single_image_kp_std /= scale_trans[batch_i, 0]



            keypoints = {kp_type: single_label[i, :] for i, kp_type in enumerate(episode_generator.support_kp_categories) \
                         if kp_mask[batch_i, i] != 0 }

            print(episode_generator.queries[batch_i])
            print(keypoints)

            im_uint8_bgr = np.zeros(single_image.shape, np.uint8)
            im_uint8_bgr[:, :, :] = single_image[:, :, ::-1]  # rgb to bgr
            # cv2.circle(im_uint8_bgr, (160, 169), 15, [255,255,255])
            # plt.imshow(im_uint8_bgr[:,:,::-1])
            # plt.show()
            color_red, color_blue, color_white, color_black,color_gray = (0, 0, 255), (255, 0, 0), (255, 255, 255), (0,0,0), (100,100,100)
            labeled_image = im_uint8_bgr
            if draw_kp_var:
                labeled_image_cur = np.copy(labeled_image)
                labeled_image_cur2 = np.copy(labeled_image)
                for kp_j in range(N):
                    if kp_mask[batch_i, kp_j] == 0:
                        continue
                    cx, cy = single_label[kp_j, 0], single_label[kp_j, 1]
                    center = (int(cx), int(cy))
                    semi_major = confidence_scale * single_image_kp_std[kp_j, 0] # + r_pixels  # std(x)  or std_major
                    semi_minor = confidence_scale * single_image_kp_std[kp_j, 1] # + r_pixels  # std(y)  or std_minor
                    semi_axes = (int(semi_major+0.5), int(semi_minor+0.5))
                    angle = 0  # independent var
                    if single_image_kp_angle is not None:  # covar
                        angle = int(single_image_kp_angle[kp_j] + 0.5)
                    kp_id = KEYPOINT_TYPES.index(episode_generator.support_kp_categories[kp_j])
                    color = datasets.dataset_utils.CocoColors[kp_id]
                    # 1) filled ellipse
                    cv2.ellipse(labeled_image_cur, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_red, thickness=ucellipse_thickness)
                    # 2) blue ellipse
                    # cv2.ellipse(labeled_image_cur2, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_white, thickness=4)  # white ellipse
                    # cv2.ellipse(labeled_image_cur2, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_red, thickness=ucellipse_thickness)
                    cv2.ellipse(labeled_image_cur2, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_blue, thickness=2)
                # 1) filled ellipse
                labeled_image_cur = cv2.addWeighted(labeled_image, 1-ucellipse_alpha, labeled_image_cur, ucellipse_alpha, 0)
                # 2) blue ellipse
                alpha2 = 1
                labeled_image_cur2 = cv2.addWeighted(labeled_image, 1-alpha2, labeled_image_cur2, alpha2, 0)

                # draw ellipse second time
                labeled_image_cur3 = np.copy(labeled_image_cur)
                for kp_j in range(N):
                    if kp_mask[batch_i, kp_j] == 0:
                        continue
                    cx, cy = single_label[kp_j, 0], single_label[kp_j, 1]
                    center = (int(cx), int(cy))
                    semi_major = confidence_scale * single_image_kp_std[kp_j, 0] # + r_pixels  # std(x)  or std_major
                    semi_minor = confidence_scale * single_image_kp_std[kp_j, 1] # + r_pixels  # std(y)  or std_minor
                    semi_axes = (int(semi_major+0.5), int(semi_minor+0.5))
                    angle = 0  # independent var
                    if single_image_kp_angle is not None:  # covar
                        angle = int(single_image_kp_angle[kp_j] + 0.5)
                    kp_id = KEYPOINT_TYPES.index(episode_generator.support_kp_categories[kp_j])
                    color = datasets.dataset_utils.CocoColors[kp_id]
                    cv2.ellipse(labeled_image_cur3, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_white, thickness=6)
                    cv2.ellipse(labeled_image_cur3, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_red, thickness=2)

                # draw predictions merely and save
                labeled_image_cur2_2= np.copy(labeled_image_cur2)
                labeled_image_cur2_2 = draw_skeletons(labeled_image_cur2_2, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}b.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur2_2)
                labeled_image_cur3_2 = np.copy(labeled_image_cur3)
                labeled_image_cur3_2 = draw_skeletons(labeled_image_cur3_2, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}c.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur3_2)
                labeled_image_cur1_2 = np.copy(labeled_image_cur)
                labeled_image_cur1_2 = draw_skeletons(labeled_image_cur1_2, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur1_2)
                # ----------------------------
                # draw gt kps and indicate whether they are bounded by uncertainty
                if kp_labels_gt is not None:
                    keypoints_gt = {kp_type: kp_labels_gt[batch_i, i, :] for i, kp_type in enumerate(episode_generator.support_kp_categories) if kp_mask[batch_i, i] != 0 }
                    # labeled_image_cur2 = draw_markers(labeled_image_cur2, keypoints_gt, marker='circle', color=color_white, circle_radius=4, thickness=-1)
                    # labeled_image_cur2 = draw_skeletons(labeled_image_cur2, [keypoints_gt], KEYPOINT_TYPES, marker='circle', circle_radius=2, thickness=-1, limbs=[], alpha=1)
                    #
                    # labeled_image_cur3 = draw_markers(labeled_image_cur3, keypoints_gt, marker='circle', color=color_white, circle_radius=4, thickness=-1)
                    # labeled_image_cur3 = draw_skeletons(labeled_image_cur3, [keypoints_gt], KEYPOINT_TYPES, marker='circle', circle_radius=2, thickness=-1, limbs=[], alpha=1)
                # ----------------------------

                # labeled_image_cur2 = draw_skeletons(labeled_image_cur2, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                # labeled_image_cur3 = draw_skeletons(labeled_image_cur3, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)

                # draw line between predictions and gt
                if kp_labels_gt is not None:
                    if os.path.exists(save_image_root + "/" + folder_name + "b") == False:
                        os.mkdir(save_image_root + "/" + folder_name + "b")
                    for kp_j, kp_type in enumerate(keypoints.keys()):  # keypoints have been filtered
                        kp_id = KEYPOINT_TYPES.index(kp_type)
                        color = datasets.dataset_utils.CocoColors[kp_id]
                        x1, y1 = int(keypoints[kp_type][0]), int(keypoints[kp_type][1])
                        x2, y2 = int(keypoints_gt[kp_type][0]), int(keypoints_gt[kp_type][1])
                        # cv2.line(labeled_image_cur2, (x1, y1), (x2, y2), color=color, thickness=1)
                        # cv2.line(labeled_image_cur3, (x1, y1), (x2, y2), color=color, thickness=1)
                        cv2.arrowedLine(labeled_image_cur2, (x1, y1), (x2, y2), color=color, thickness=2, tipLength=None)
                        cv2.arrowedLine(labeled_image_cur3, (x1, y1), (x2, y2), color=color_white, thickness=6, tipLength=None)
                        cv2.arrowedLine(labeled_image_cur3, (x1, y1), (x2, y2), color=color, thickness=2, tipLength=None)
                    cv2.imwrite(os.path.join(save_image_root, folder_name+"b", '{}_q_{}b.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur2)
                    cv2.imwrite(os.path.join(save_image_root, folder_name+"b", '{}_q_{}c.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur3)

                #     polygon = cv2.ellipse2Poly(center, semi_axes, angle, 0, 360, 1)
                #     cv2.fillConvexPoly(labeled_image, polygon, (0, 0, 0))
                # labeled_image = cv2.addWeighted(labeled_image_pre, 0.7, labeled_image, 0.3, 0)

                labeled_image = labeled_image_cur

            if draw_kp_var == False:  # only used in drawing GT kp labels
                # circle 8; tilted_cross: 6
                r_pixels = 10
                labeled_image = draw_skeletons(labeled_image, [keypoints], KEYPOINT_TYPES, marker='circle', circle_radius=r_pixels, limbs=[], alpha=0.7)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image)
                labeled_image = draw_markers(labeled_image, keypoints, marker='circle', color=[255,255,255], circle_radius=r_pixels, thickness=2)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}b.jpg'.format(save_file_prefix, batch_i)), labeled_image)
            else:

                # ----------------------------
                # draw gt kps and indicate whether they are bounded by uncertainty
                # if kp_labels_gt is not None:
                #     labeled_image = draw_markers(labeled_image, keypoints_gt, marker='circle', color=[255,255,255], circle_radius=4, thickness=-1)
                #     labeled_image = draw_skeletons(labeled_image, [keypoints_gt], KEYPOINT_TYPES, marker='circle', circle_radius=2, thickness=-1, limbs=[], alpha=1)
                    # labeled_image2 = draw_markers(labeled_image2, keypoints_gt, marker='circle', color=[0,0,0], circle_radius=4, thickness=1)
                # ----------------------------
                # labeled_image = draw_skeletons(labeled_image, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                pass





            labeled_image = labeled_image[:, :, ::-1]  # bgr to rgb


            if is_show:
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(single_image)
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(labeled_image)
                plt.show()

            if is_save:
                # write labelled images
                # cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image[:,:,::-1])

                # draw line between predictions and gt
                if kp_labels_gt is not None:
                    labeled_image = labeled_image[:, :, ::-1]
                    if os.path.exists(save_image_root + "/" + folder_name + "b") == False:
                        os.mkdir(save_image_root + "/" + folder_name + "b")
                    for kp_j, kp_type in enumerate(keypoints.keys()):  # keypoints have been filtered
                        kp_id = KEYPOINT_TYPES.index(kp_type)
                        color = datasets.dataset_utils.CocoColors[kp_id]
                        x1, y1 = int(keypoints[kp_type][0]), int(keypoints[kp_type][1])
                        x2, y2 = int(keypoints_gt[kp_type][0]), int(keypoints_gt[kp_type][1])
                        # cv2.line(labeled_image, (x1, y1), (x2, y2), color=color, thickness=1)
                        cv2.arrowedLine(labeled_image, (x1, y1), (x2, y2), color=color, thickness=2, tipLength=None)
                    cv2.imwrite(os.path.join(save_image_root, folder_name+"b", '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image)

def show_save_predictions(images: torch.Tensor, kp_labels: torch.Tensor, kp_mask: torch.Tensor, scale_trans: torch.Tensor, episode_generator: EpisodeGenerator,
                          square_image_length=368, kp_var=None, confidence_scale=1, version='preprocessed', is_show=False, is_save=True, folder_name='query_gt', save_file_prefix="", kp_labels_gt=None, root_postfix=""):
    # queries: B2 x C x H x W
    # query_labels: B2 x N x 2
    # scale_trans: B2 x 6 (scale, x_offset, y_offset, bbx_area, pad_xoffset, pad_yoffset)
    # kp_var: B2 x N x 2 (var_x, var_y) or covariance B2 x N x 2 x 2
    # confidence_scale: confidence_scale * std (like 3*sigma)
    import copy
    images, kp_labels = copy.deepcopy(images.cpu().detach()), copy.deepcopy(kp_labels)
    kp_mask, scale_trans = copy.deepcopy(kp_mask), copy.deepcopy(scale_trans)
    draw_kp_var = True if kp_var is not None else False
    if draw_kp_var:
        # if not None, format is independent var B2 x N x 2, [var(x), var(y)]
        # or covariance B2 x N x 2 x 2, Covar = [Sigma_xx, Sigma_xy; Sigma_xy, Sigma_yy]
        kp_var = copy.deepcopy(kp_var)

    if kp_labels_gt is not None:
        kp_labels_gt = copy.deepcopy(kp_labels_gt)
        kp_labels_gt = recover_kps(kp_labels_gt, square_image_length, scale_trans)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    r_pixels = 9  # circle 8; tilted_cross: 6/8/9/10
    ucellipse_thickness = -1  # thickness default: 1 or 2; fill ellipse: -1
    ucellipse_alpha = 0.32    # opaque, 1 ; fill ellipse: 0.4/0.35/0.32

    save_image_root = './episode_images/predictions'+root_postfix  # root_postfix is used to distinguish diff. datasets
    if os.path.exists(save_image_root) == False:
        os.makedirs(save_image_root)
    if os.path.exists(save_image_root + "/" + folder_name) == False:
        os.makedirs(save_image_root + "/" + folder_name)

    B = kp_labels.shape[0]  # number of query image
    N = kp_labels.shape[1]  # number of keypoints
    if version == 'preprocessed':  # show keypoints in pre-processed images
        for batch_i in range(B):
            single_image, single_label = images[batch_i, :, :, :], kp_labels[batch_i, :, :]
            single_image = single_image.permute(1, 2, 0)  # H x W x C

            # single_label = (single_label * (square_image_length-1)).long()
            single_label = ((single_label / 2 + 0.5) * (square_image_length - 1)).long()
            keypoints = {kp_type: single_label[i, :] for i, kp_type in enumerate(episode_generator.support_kp_categories) \
                         if kp_mask[batch_i, i] != 0 }

            print(episode_generator.queries[batch_i])
            print(keypoints)

            # support_image ranges 0~1
            for c in range(3):
                single_image[:, :, c] = single_image[:, :, c] * std[c] + mean[c]
            im_uint8 = single_image.mul(255).numpy().astype(np.uint8)
            im_uint8_bgr = np.zeros(im_uint8.shape, np.uint8)
            im_uint8_bgr[:, :, :] = im_uint8[:, :, ::-1]  # rgb to bgr
            # cv2.circle(im_uint8_bgr, (160, 169), 15, [255,255,255])
            # plt.imshow(im_uint8_bgr[:,:,::-1])
            # plt.show()

            # labeled_image = draw_skeletons(im_uint8_bgr, [keypoints], KEYPOINT_TYPES, marker='circle', circle_radius=r_pixels, limbs=[], alpha=0.7)
            labeled_image = draw_skeletons(im_uint8_bgr, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
            labeled_image = labeled_image[:, :, ::-1]  # bgr to rgb

            if is_show:
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(single_image)
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(labeled_image)
                plt.show()

            if is_save:
                # write labelled images
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image[:,:,::-1])
    else:  # show keypoints in original image
        import PIL.Image as Image
        # uncertainty_e
        for batch_i in range(B):
            single_label = kp_labels[batch_i, :, :]
            single_image = Image.open(episode_generator.queries[batch_i]['filename']).convert('RGB')  # H x W x C
            single_image = np.array(single_image)  # .astype(np.uint8)

            # transform the scale-padded image annotations to original annotations
            # single_label = (single_label * (square_image_length-1)).long()
            single_label = single_label / 2 + 0.5  # 0~1
            single_label *= square_image_length - 1
            single_label += scale_trans[batch_i, 1:3].repeat(N, 1)
            single_label /= scale_trans[batch_i, 0]
            single_label = single_label.long()

            if draw_kp_var:
                if len(kp_var.shape) == 3:  # independent var B2 x N x 2, [var(x), var(y)]
                    single_image_kp_var = kp_var[batch_i, :, :]
                    single_image_kp_std = torch.sqrt(single_image_kp_var)
                    single_image_kp_std /= scale_trans[batch_i, 0]  # N x 2
                    single_image_kp_angle = None
                elif len(kp_var.shape) == 4: # covariance B2 x N x 2 x 2, Covar = [Sigma_xx, Sigma_xy; Sigma_xy, Sigma_yy]
                    single_image_kp_covar = kp_var[batch_i, :, :, :]  # N x 2 x 2
                    single_image_kp_std = torch.zeros(N, 2)  # N x 2
                    single_image_kp_angle = torch.zeros(N)   # N
                    for kp_j in range(N):
                        e, _, angle = compute_eigenvalues(single_image_kp_covar[kp_j])
                        single_image_kp_std[kp_j, :] = torch.sqrt(e[:, 0])  # sqrt(major_axis, minor_axis)
                        single_image_kp_angle[kp_j] = angle

                    single_image_kp_std /= scale_trans[batch_i, 0]



            keypoints = {kp_type: single_label[i, :] for i, kp_type in enumerate(episode_generator.support_kp_categories) \
                         if kp_mask[batch_i, i] != 0 }

            print(episode_generator.queries[batch_i])
            print(keypoints)

            im_uint8_bgr = np.zeros(single_image.shape, np.uint8)
            im_uint8_bgr[:, :, :] = single_image[:, :, ::-1]  # rgb to bgr
            # cv2.circle(im_uint8_bgr, (160, 169), 15, [255,255,255])
            # plt.imshow(im_uint8_bgr[:,:,::-1])
            # plt.show()
            color_red, color_blue, color_white, color_black,color_gray = (0, 0, 255), (255, 0, 0), (255, 255, 255), (0,0,0), (100,100,100)
            color_purple = (255, 0, 255)
            dark_red = (0,0,232)
            light_green, cyan, orange = (128, 255,128), (255,255,0), (64,128,255)
            labeled_image = im_uint8_bgr
            if draw_kp_var:
                labeled_image_cur = np.copy(labeled_image)
                labeled_image_cur2 = np.copy(labeled_image)
                for kp_j in range(N):
                    if kp_mask[batch_i, kp_j] == 0:
                        continue
                    cx, cy = single_label[kp_j, 0], single_label[kp_j, 1]
                    center = (int(cx), int(cy))
                    semi_major = confidence_scale * single_image_kp_std[kp_j, 0] # + r_pixels  # std(x)  or std_major
                    semi_minor = confidence_scale * single_image_kp_std[kp_j, 1] # + r_pixels  # std(y)  or std_minor
                    semi_axes = (int(semi_major+0.5), int(semi_minor+0.5))
                    angle = 0  # independent var
                    if single_image_kp_angle is not None:  # covar
                        angle = int(single_image_kp_angle[kp_j] + 0.5)
                    kp_id = KEYPOINT_TYPES.index(episode_generator.support_kp_categories[kp_j])
                    color = datasets.dataset_utils.CocoColors[kp_id]
                    # 1) filled ellipse
                    cv2.ellipse(labeled_image_cur, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_red, thickness=ucellipse_thickness)
                    # 2) blue ellipse
                    # cv2.ellipse(labeled_image_cur2, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_white, thickness=4)  # white ellipse
                    # cv2.ellipse(labeled_image_cur2, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_red, thickness=ucellipse_thickness)
                    cv2.ellipse(labeled_image_cur2, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_blue, thickness=2)
                # 1) filled ellipse
                labeled_image_cur = cv2.addWeighted(labeled_image, 1-ucellipse_alpha, labeled_image_cur, ucellipse_alpha, 0)
                # 2) blue ellipse
                alpha2 = 1
                labeled_image_cur2 = cv2.addWeighted(labeled_image, 1-alpha2, labeled_image_cur2, alpha2, 0)

                # draw ellipse second time
                labeled_image_cur3 = np.copy(labeled_image_cur)
                for kp_j in range(N):
                    if kp_mask[batch_i, kp_j] == 0:
                        continue
                    cx, cy = single_label[kp_j, 0], single_label[kp_j, 1]
                    center = (int(cx), int(cy))
                    semi_major = confidence_scale * single_image_kp_std[kp_j, 0] # + r_pixels  # std(x)  or std_major
                    semi_minor = confidence_scale * single_image_kp_std[kp_j, 1] # + r_pixels  # std(y)  or std_minor
                    semi_axes = (int(semi_major+0.5), int(semi_minor+0.5))
                    angle = 0  # independent var
                    if single_image_kp_angle is not None:  # covar
                        angle = int(single_image_kp_angle[kp_j] + 0.5)
                    kp_id = KEYPOINT_TYPES.index(episode_generator.support_kp_categories[kp_j])
                    color = datasets.dataset_utils.CocoColors[kp_id]
                    # cv2.ellipse(labeled_image_cur3, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_white, thickness=6)
                    cv2.ellipse(labeled_image_cur3, center, semi_axes, angle=angle, startAngle=0, endAngle=360, color=color_red, thickness=2)

                # draw predictions merely and save
                labeled_image_cur2_2= np.copy(labeled_image_cur2)
                labeled_image_cur2_2 = draw_skeletons(labeled_image_cur2_2, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}b.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur2_2)
                labeled_image_cur3_2 = np.copy(labeled_image_cur3)
                labeled_image_cur3_2 = draw_skeletons(labeled_image_cur3_2, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}c.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur3_2)
                labeled_image_cur1_2 = np.copy(labeled_image_cur)
                labeled_image_cur1_2 = draw_skeletons(labeled_image_cur1_2, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur1_2)
                # ----------------------------
                # draw gt kps and indicate whether they are bounded by uncertainty
                if kp_labels_gt is not None:
                    keypoints_gt = {kp_type: kp_labels_gt[batch_i, i, :] for i, kp_type in enumerate(episode_generator.support_kp_categories) if kp_mask[batch_i, i] != 0 }
                    labeled_image_cur2 = draw_markers(labeled_image_cur2, keypoints_gt, marker='circle', color=color_white, circle_radius=4, thickness=-1)
                    labeled_image_cur2 = draw_skeletons(labeled_image_cur2, [keypoints_gt], KEYPOINT_TYPES, marker='circle', circle_radius=2, thickness=-1, limbs=[], alpha=1)

                    labeled_image_cur3 = draw_markers(labeled_image_cur3, keypoints_gt, marker='circle', color=color_white, circle_radius=4, thickness=-1)
                    labeled_image_cur3 = draw_skeletons(labeled_image_cur3, [keypoints_gt], KEYPOINT_TYPES, marker='circle', circle_radius=2, thickness=-1, limbs=[], alpha=1)
                # ----------------------------
                labeled_image_cur2 = draw_skeletons(labeled_image_cur2, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                labeled_image_cur3 = draw_skeletons(labeled_image_cur3, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)

                # draw line between predictions and gt
                if kp_labels_gt is not None:
                    if os.path.exists(save_image_root + "/" + folder_name + "b") == False:
                        os.mkdir(save_image_root + "/" + folder_name + "b")
                    for kp_j, kp_type in enumerate(keypoints.keys()):  # keypoints have been filtered
                        kp_id = KEYPOINT_TYPES.index(kp_type)
                        color = datasets.dataset_utils.CocoColors[kp_id]
                        x1, y1 = int(keypoints[kp_type][0]), int(keypoints[kp_type][1])
                        x2, y2 = int(keypoints_gt[kp_type][0]), int(keypoints_gt[kp_type][1])
                        cv2.line(labeled_image_cur2, (x1, y1), (x2, y2), color=color, thickness=2)
                        cv2.line(labeled_image_cur3, (x1, y1), (x2, y2), color=color, thickness=2)
                        # cv2.arrowedLine(labeled_image_cur2, (x1, y1), (x2, y2), color=color, thickness=2, tipLength=None)
                        # cv2.arrowedLine(labeled_image_cur3, (x1, y1), (x2, y2), color=color_white, thickness=6, tipLength=None)
                        # cv2.arrowedLine(labeled_image_cur3, (x1, y1), (x2, y2), color=color, thickness=2, tipLength=None)
                    cv2.imwrite(os.path.join(save_image_root, folder_name+"b", '{}_q_{}b.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur2)
                    cv2.imwrite(os.path.join(save_image_root, folder_name+"b", '{}_q_{}c.jpg'.format(save_file_prefix, batch_i)), labeled_image_cur3)

                #     polygon = cv2.ellipse2Poly(center, semi_axes, angle, 0, 360, 1)
                #     cv2.fillConvexPoly(labeled_image, polygon, (0, 0, 0))
                # labeled_image = cv2.addWeighted(labeled_image_pre, 0.7, labeled_image, 0.3, 0)

                labeled_image = labeled_image_cur

            if draw_kp_var == False:  # only used in drawing GT kp labels
                # circle 8; tilted_cross: 6
                r_pixels = 10
                labeled_image = draw_skeletons(labeled_image, [keypoints], KEYPOINT_TYPES, marker='circle', circle_radius=r_pixels, limbs=[], alpha=0.7)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image)
                labeled_image = draw_markers(labeled_image, keypoints, marker='circle', color=[255,255,255], circle_radius=r_pixels, thickness=2)
                cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}b.jpg'.format(save_file_prefix, batch_i)), labeled_image)
            else:

                # ----------------------------
                # draw gt kps and indicate whether they are bounded by uncertainty
                if kp_labels_gt is not None:
                    labeled_image = draw_markers(labeled_image, keypoints_gt, marker='circle', color=[255,255,255], circle_radius=4, thickness=-1)
                    labeled_image = draw_skeletons(labeled_image, [keypoints_gt], KEYPOINT_TYPES, marker='circle', circle_radius=2, thickness=-1, limbs=[], alpha=1)
                    # labeled_image2 = draw_markers(labeled_image2, keypoints_gt, marker='circle', color=[0,0,0], circle_radius=4, thickness=1)
                # ----------------------------
                labeled_image = draw_skeletons(labeled_image, [keypoints], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=r_pixels, limbs=[], alpha=1)
                pass





            labeled_image = labeled_image[:, :, ::-1]  # bgr to rgb


            if is_show:
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(single_image)
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(labeled_image)
                plt.show()

            if is_save:
                # write labelled images
                # cv2.imwrite(os.path.join(save_image_root, folder_name, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image[:,:,::-1])

                # draw line between predictions and gt
                if kp_labels_gt is not None:
                    labeled_image = labeled_image[:, :, ::-1]
                    if os.path.exists(save_image_root + "/" + folder_name + "b") == False:
                        os.mkdir(save_image_root + "/" + folder_name + "b")
                    for kp_j, kp_type in enumerate(keypoints.keys()):  # keypoints have been filtered
                        kp_id = KEYPOINT_TYPES.index(kp_type)
                        color = datasets.dataset_utils.CocoColors[kp_id]
                        x1, y1 = int(keypoints[kp_type][0]), int(keypoints[kp_type][1])
                        x2, y2 = int(keypoints_gt[kp_type][0]), int(keypoints_gt[kp_type][1])
                        cv2.line(labeled_image, (x1, y1), (x2, y2), color=color, thickness=2)
                        # cv2.arrowedLine(labeled_image, (x1, y1), (x2, y2), color=color, thickness=2, tipLength=None)
                    cv2.imwrite(os.path.join(save_image_root, folder_name+"b", '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), labeled_image)


def preprocess_uncertainty(src_pts, dst_pts, uncertainty):
    from tps_warp_uncertainty_weighted import TPS

    d = np.sqrt(((dst_pts - src_pts) ** 2).sum(-1))  # N
    d_max = np.max(d)
    d_mean = np.mean(d)
    d_med = np.median(d)
    d_p = np.percentile(d, 80, axis=0)
    l = 5 # 2 * (d_max-d_mean)
    lambd = TPS.u(l)
    beta = 2
    # uncertainty = np.array([1, 5, 1, 1], dtype=np.float32)
    print('l: %f, lambd: %f, beta: %f'%(l, lambd, beta))

    # print(uncertainty)
    # D = 1 / uncertainty
    # print(D)
    # D = D / sum(D)
    # D = np.power(D, beta)
    # D = D / sum(D)
    # D = 1 / D
    # print(D)

    s = sum(uncertainty)
    D = uncertainty / s
    D = np.power(D, beta)
    D = D / sum(D)
    D = s * D
    print(D)

    return lambd, D

def draw_kps_lines(image, src_kps, tgt_kps, kp_mask, support_kp_types):
    '''
    src_kps: K valid kps
    kp_mask: N (N >= K)
    support_kp_types: list, N
    '''
    from collections import OrderedDict
    src_skeleton, tgt_skeleton = OrderedDict(), OrderedDict()
    cnt = 0
    for i, kp_type in enumerate(support_kp_types):
        if kp_mask[i] == 1:
            src_skeleton[kp_type] = src_kps[cnt]
            tgt_skeleton[kp_type] = tgt_kps[cnt]
            cnt += 1
    image = draw_skeletons(image, [src_skeleton], KEYPOINT_TYPES, marker='tilted_cross', circle_radius=6, alpha=1)
    image = draw_skeletons(image, [tgt_skeleton], KEYPOINT_TYPES, marker='circle', circle_radius=6, alpha=1, thickness=1)

    for kp_type in src_skeleton.keys():
        kp_id = KEYPOINT_TYPES.index(kp_type)
        color = datasets.dataset_utils.CocoColors[kp_id]
        x1, y1 = int(src_skeleton[kp_type][0]), int(src_skeleton[kp_type][1])
        x2, y2 = int(tgt_skeleton[kp_type][0]), int(tgt_skeleton[kp_type][1])
        cv2.line(image, (x1, y1), (x2, y2), color=color, thickness=1)

    return image

def show_save_warp_images(images: torch.Tensor, gt_kps: torch.Tensor, predict_kps: torch.Tensor, kp_mask: torch.Tensor, scale_trans: torch.Tensor, ref_kps: torch.Tensor, ref_scale_trans: torch.Tensor,
                          episode_generator: EpisodeGenerator, square_image_length=368, kp_var=None, confidence_scale=1, version='preprocessed', is_show=False, is_save=True, save_file_prefix=""):
    # queries: B2 x C x H x W
    # gt_kps / predict_kps: B2 x N x 2
    # kp_mask: B2 x N, it indicates both query keypoint and ref keypoint for specific body part are valid
    # scale_trans: B2 x 6 (scale, x_offset, y_offset, bbx_area, pad_xoffset, pad_yoffset)
    # ref_kps: B1 x N x 2 (support kps, if there are multiple support images, we need to select one image as ref image)
    import copy
    from TPSwarping import WarpImage_TPS
    from tps_warp_uncertainty_weighted import warp_image, TPS
    import PIL.Image as Image

    images, gt_kps, predict_kps = copy.deepcopy(images.cpu().detach()), copy.deepcopy(gt_kps), copy.deepcopy(predict_kps)
    kp_mask, scale_trans = copy.deepcopy(kp_mask).bool(), copy.deepcopy(scale_trans)
    ref_kps, ref_scale_trans = copy.deepcopy(ref_kps), copy.deepcopy(ref_scale_trans)
    use_kp_var = True if kp_var is not None else False
    if use_kp_var:
        # if not None, format is independent var B2 x N x 2, [var(x), var(y)]
        # or covariance B2 x N x 2 x 2, Covar = [Sigma_xx, Sigma_xy; Sigma_xy, Sigma_yy]
        kp_var = copy.deepcopy(kp_var)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    r_pixels = 8

    if version == 'original':
        save_image_root = './episode_images/warp_origin_images'
    else:  # == 'preprocessed'
        save_image_root = './episode_images/warp_prepro_images'
    folder_qualified_kps = "qualified_pred_kps"
    folder_all_kps = "all_pred_kps"
    folder_gt_kps = "gt"
    folder_uc = "all_pred_uc"
    if os.path.exists(save_image_root) == False:
        os.mkdir(save_image_root)
    if os.path.exists(save_image_root + "/" + folder_qualified_kps) == False:
        os.mkdir(save_image_root + "/" + folder_qualified_kps)
    if os.path.exists(save_image_root + "/" + folder_all_kps) == False:
        os.mkdir(save_image_root + "/" + folder_all_kps)
    if os.path.exists(save_image_root + "/" + folder_gt_kps) == False:
        os.mkdir(save_image_root + "/" + folder_gt_kps)
    if os.path.exists(save_image_root + "/" + folder_uc) == False:
        os.mkdir(save_image_root + "/" + folder_uc)

    if version == 'original':
        # transform the scale-padded image annotations to original annotations
        gt_kps = recover_kps(gt_kps, square_image_length, scale_trans)
        predict_kps = recover_kps(predict_kps, square_image_length, scale_trans)
        ref_kps = recover_kps(ref_kps, square_image_length, ref_scale_trans)
    elif version == 'preprocessed':
        # transform -1~1 to preprocessed image size
        # gt_kps = (gt_kps * (square_image_length-1)).long()
        gt_kps = ((gt_kps / 2 + 0.5) * (square_image_length - 1)).long()
        predict_kps = ((predict_kps / 2 + 0.5) * (square_image_length - 1)).long()
        ref_kps = ((ref_kps / 2 + 0.5) * (square_image_length - 1)).long()

    B1 = ref_kps.shape[0]  # number of support image
    B2 = gt_kps.shape[0]  # number of query image
    N = gt_kps.shape[1]  # number of keypoints
    if B1 > 1:  # choose one support image as reference image for warping
        ref_kps_choosed = ref_kps[0, :, :]  # N x 2, choose first image
    else:
        ref_kps_choosed = ref_kps.reshape(N, 2)
    pck_thresh_img = np.array([0.06, 0.1]) # np.array([20.0 / 368, 40.0 / 368])

    # compute uncertainty energy for uncertainty-weighted warp
    if use_kp_var:
        kp_stds = torch.zeros(B2, N, 2)
        kp_uncertainty_energy = torch.zeros(B2, N)  # 9*(std1^2 + std2^2), confidence_scale=3
        kp_uncertainty_energy2 = torch.zeros(B2, N)
        for batch_i in range(B2):
            if len(kp_var.shape) == 3:  # independent var B2 x N x 2, [var(x), var(y)]
                single_image_kp_var = kp_var[batch_i, :, :]
                single_image_kp_std = torch.sqrt(single_image_kp_var)
                single_image_kp_std /= scale_trans[batch_i, 0]  # N x 2
                single_image_kp_angle = None
                kp_stds[batch_i, :, :] = single_image_kp_std
                kp_uncertainty_energy[batch_i, :] = torch.sum((confidence_scale*single_image_kp_std) ** 2, dim=1)

            elif len(kp_var.shape) == 4: # covariance B2 x N x 2 x 2, Covar = [Sigma_xx, Sigma_xy; Sigma_xy, Sigma_yy]
                single_image_kp_covar = kp_var[batch_i, :, :, :]  # N x 2 x 2
                single_image_kp_std = torch.zeros(N, 2)  # N x 2
                single_image_kp_angle = torch.zeros(N)   # N
                for kp_j in range(N):
                    e, _, angle = compute_eigenvalues(single_image_kp_covar[kp_j])
                    single_image_kp_std[kp_j, :] = torch.sqrt(e[:, 0])  # sqrt(major_axis, minor_axis)
                    single_image_kp_angle[kp_j] = angle

                    kp_uncertainty_energy2[batch_i, kp_j] = torch.trace(single_image_kp_covar[kp_j]) * (confidence_scale/scale_trans[batch_i, 0]) ** 2

                single_image_kp_std /= scale_trans[batch_i, 0]
                kp_stds[batch_i, :, :] = single_image_kp_std
                kp_uncertainty_energy[batch_i, :] = torch.sum((confidence_scale*single_image_kp_std) ** 2, dim=1)


    for batch_i in range(B2):
        single_gt_kps, single_pred_kps = gt_kps[batch_i, :, :], predict_kps[batch_i, :, :]

        if version == 'preprocessed':  # warp on pre-processed images
            single_image= images[batch_i, :, :, :]
            single_image = single_image.permute(1, 2, 0)  # H x W x C
            # support_image ranges 0~1
            for c in range(3):
                single_image[:, :, c] = single_image[:, :, c] * std[c] + mean[c]
            # center padding
            left = scale_trans[batch_i, 4].long()
            top = scale_trans[batch_i, 5].long()
            if left == 0 and -top > 0:
                single_image[0:-top, :, :] = 0
                single_image[square_image_length + top:square_image_length, :, :] = 0
            elif top == 0 and -left > 0:
                single_image[:, 0:-left, :] = 0
                single_image[:, square_image_length + left:square_image_length, :] = 0
            # single_query_image = single_query_image[:, -top:square_image_length + top, -left:square_image_length + left]  # crop
            im_uint8 = single_image.mul(255).numpy().astype(np.uint8)  # rgb
        else: # 'original', warp on original image
            single_image = Image.open(episode_generator.queries[batch_i]['filename']).convert('RGB')  # H x W x C
            im_uint8 = np.array(single_image)  # .astype(np.uint8)

        im_uint8 = im_uint8[:, :, [2, 1, 0]]  # rgb to bgr, for appropriately drawing using opencv functions (important)
        h, w = im_uint8.shape[0:2]
        diagonal_length = torch.tensor(np.sqrt(h**2 + w**2))
        distance_thresh = (pck_thresh_img[0] * diagonal_length)
        if torch.sum(kp_mask[batch_i]) < 4:
            continue
        tgt_gt_kps = ref_kps_choosed[kp_mask[batch_i], :]    # k x 2, k valid target ref kps
        src_gt_kps = single_gt_kps[kp_mask[batch_i], :]      # k x 2, k valid source gt kps
        src_pred_kps = single_pred_kps[kp_mask[batch_i], :]  # k x 2, k valid source predict kps

        qualified_mask = torch.sum((src_pred_kps - src_gt_kps) ** 2, dim=1) < distance_thresh ** 2  # k
        src_qualified_pred_kps = src_pred_kps[qualified_mask]  # kk x 2, kk accurate predict kps
        tgt_qualified_gt_kps = tgt_gt_kps[qualified_mask]      # kk x 2, kk target corresponding kps
        # 1) src_qualified_pred_kps <---> tgt_qualified_gt_kps
        # if torch.sum(qualified_mask) >= 4:  # needs at least four pairs of corresponding kps
        #     src_qualified_pred_kps = src_qualified_pred_kps.numpy().astype(np.float32)
        #     tgt_qualified_gt_kps   = tgt_qualified_gt_kps.numpy().astype(np.float32)
        #     # new_im, new_src_pts, new_tgt_pts = WarpImage_TPS(src_qualified_pred_kps, tgt_qualified_gt_kps, im_uint8)
        #     new_im, new_src_pts = warp_image(im_uint8, src_qualified_pred_kps, tgt_qualified_gt_kps, im_uint8.shape[:2], unit_space=False, lambd=0, uncertainty_square=None)
        #     if is_save:
        #         cv2.imwrite(os.path.join(save_image_root, folder_qualified_kps, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), new_im)  # rgb to bgr
        # 2) src_pred_kps <---> tgt_gt_kps
        src_pred_kps = src_pred_kps.numpy().astype(np.float32)
        tgt_gt_kps = tgt_gt_kps.numpy().astype(np.float32)
        # new_im2, new_src_pts2, new_tgt_pts2 = WarpImage_TPS(src_pred_kps, tgt_gt_kps, im_uint8)
        new_im2, new_src_pts2 = warp_image(im_uint8, src_pred_kps, tgt_gt_kps, im_uint8.shape[:2], unit_space=False, lambd=0, uncertainty_square=None)
        if new_im2 is None:  # singular matrix error
            continue
        new_im2 = draw_kps_lines(new_im2, new_src_pts2, tgt_gt_kps, kp_mask[batch_i], episode_generator.support_kp_categories)
        if is_save:
            # cv2.imwrite(os.path.join(save_image_root, folder_all_kps, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), new_im2[:,:,::-1])  # rgb to bgr
            cv2.imwrite(os.path.join(save_image_root, folder_uc, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), new_im2)  # rgb to bgr
        # 3) src_gt_kps <---> tgt_gt_kps
        src_gt_kps = src_gt_kps.numpy().astype(np.float32)
        # new_im3, new_src_pts3, new_tgt_pts3 = WarpImage_TPS(src_gt_kps, tgt_gt_kps, im_uint8)
        new_im3, new_src_pts3 = warp_image(im_uint8, src_gt_kps, tgt_gt_kps, im_uint8.shape[:2], unit_space=False, lambd=0, uncertainty_square=None)
        new_im3 = draw_kps_lines(new_im3, new_src_pts3, tgt_gt_kps, kp_mask[batch_i], episode_generator.support_kp_categories)
        if is_save:
            cv2.imwrite(os.path.join(save_image_root, folder_gt_kps, '{}_q_{}.jpg'.format(save_file_prefix, batch_i)), new_im3)  # rgb to bgr

        # 4) src_pred_kps <---> tgt_gt_kps, using kp uncertainty energy for uncertainty-weighted tps warp
        if use_kp_var:
            single_pred_uc = kp_uncertainty_energy[batch_i, :]
            src_pred_uc = single_pred_uc[kp_mask[batch_i]]      # k, k valid uncertainty energy for source predict kps
            lambd, uncertainty_sq = preprocess_uncertainty(src_pred_kps, tgt_gt_kps, src_pred_uc)
            new_im4, new_src_pts4 = warp_image(im_uint8, src_pred_kps, tgt_gt_kps, im_uint8.shape[:2], unit_space=False, lambd=lambd, uncertainty_square=uncertainty_sq)
            new_im4 = draw_kps_lines(new_im4, new_src_pts4, tgt_gt_kps, kp_mask[batch_i], episode_generator.support_kp_categories)
            # identical kp uncertainty energy for warping
            new_im5, new_src_pts5 = warp_image(im_uint8, src_pred_kps, tgt_gt_kps, im_uint8.shape[:2], unit_space=False, lambd=TPS.u(20), uncertainty_square=np.ones(sum(kp_mask[batch_i])))
            new_im5 = draw_kps_lines(new_im5, new_src_pts5, tgt_gt_kps, kp_mask[batch_i], episode_generator.support_kp_categories)
            if is_save:
                cv2.imwrite(os.path.join(save_image_root, folder_uc, '{}_q_{}b.jpg'.format(save_file_prefix, batch_i)), new_im4)  # rgb to bgr
                # identical kp uncertainty energy for warping
                cv2.imwrite(os.path.join(save_image_root, folder_uc, '{}_q_{}c.jpg'.format(save_file_prefix, batch_i)), new_im5)  # rgb to bgr


        if is_show:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(im_uint8)
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(new_im3)
            plt.show()

        print(episode_generator.queries[batch_i])

def show_cam_on_image(original_img, mask, save_path, mode='color'):
    # original_img: H x W x C, BGR format, value range 0~1
    # mask: H x W, value ranges 0~1

    # original_img.shape (height, width, channel), shape[-2::-1] == (width, height)
    # scaled_mask = cv2.resize(mask, original_img.shape[-2::-1])
    original_img = np.float32(original_img) * 255

    if mode == 'color':
        heatmap = cv2.applyColorMap(np.uint8(255*mask*0.72), cv2.COLORMAP_JET)  # set 0.72 to adjust the jet color ranges
        cam = np.float32(heatmap)*0.35 + np.float32(original_img)
    else:  # 'gray
        H, W = mask.shape
        heatmap = mask.reshape(H, W, 1).repeat(3, axis=2) * 255 * 0.72
        cam = np.float32(heatmap) * 0.7 + np.float32(original_img) * 0.3

    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))

def show_save_attention_maps(attention_maps : np.ndarray, normalized_images : torch.Tensor, support_kp_mask: torch.Tensor, query_kp_mask: torch.Tensor,
                             support_kp_categories, episode_num=0, is_show=False, is_save=True, save_image_root='./episode_images/attention_maps', delete_old=True, T=None):
    (B, C, H, W) = normalized_images.shape
    (_, N, H2, W2) = attention_maps.shape
    scale = H / H2
    normalized_images2 = normalized_images.cpu().detach().numpy().transpose(0, 2, 3, 1)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    ## Only applied softmax in valid region, if un-commenting below code we need to modify feature_modulator2/1
    # attention_maps2 = torch.zeros(attention_maps.shape)
    # for batch_i in range(B):
    #     # center padding
    #     left = int((-T[batch_i, 1].long()) / scale + 0.5)
    #     top = int((-T[batch_i, 2].long())  / scale + 0.5)
    #     right = int((368 - 1 + T[batch_i, 1].long())  / scale + 0.5)
    #     bottom = int((368 - 1 + T[batch_i, 2].long()) / scale + 0.5)
    #     attention_maps2[batch_i, :, top:bottom, left:right] = attention_maps[batch_i, :, top:bottom, left:right]  # crop
    #     T_scale = float(T[batch_i, 0])
    # attention_maps2 = attention_maps2.reshape(B, N, -1)
    # attention_maps2 = nn.functional.softmax(attention_maps2, dim=2).reshape(B, N, H2, W2)
    # attention_maps = attention_maps2.numpy()
    attention_maps = attention_maps.numpy()

    # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
    union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0  # N
    # compute the valid query keypoints, using broadcast
    valid_kp_mask = (query_kp_mask * union_support_kp_mask.reshape(1, -1))  # B2 x N
    # invisible_kp_mask = support_kp_mask - valid_kp_mask

    # save_image_root = './attention_maps'
    # if delete_old == True:
    #     for each_file in os.listdir(save_image_root):
    #         os.remove(os.path.join(save_image_root, each_file))
    if os.path.exists(save_image_root) == False:
        os.mkdir(save_image_root)
    if os.path.exists(save_image_root + '/visible_predicts') == False:
        os.mkdir(save_image_root + '/visible_predicts')
    if os.path.exists(save_image_root + '/invisible_predicts') == False:
        os.mkdir(save_image_root + '/invisible_predicts')

    # plt.ion()
    # plt.figure()
    for i in range(B):
        for j in range(N):
            if union_support_kp_mask[j] == 0:  # skip those invalid support keypoints
                continue
            image_i = np.zeros((H, W, 3), dtype=np.float32)
            attention_maps_j = np.zeros((H, W), dtype=np.float32)
            for c in range(3):
                image_i[:, :, c] = (normalized_images2[i, :, :, c] * std[c] + mean[c])
            attention_map_resized = cv2.resize(attention_maps[i, j, :, :], (W, H), interpolation=cv2.INTER_CUBIC)
            # attention_map_resized = cv2.resize(attention_maps[i, j, :, :], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # rescale the attention map according to its min value and max value
            vmin, vmax = np.min(attention_map_resized), np.max(attention_map_resized)
            # =======================
            # normalization method 1
            # if (vmax - vmin) > 0:  # vmin, vmax >= 0 since softmax applied
            #     attention_map_resized = (attention_map_resized - vmin) / (vmax - vmin)
            # =======================
            # normalization method 2, advantage: it can reflect the difference between vmin and vmax. For example, if vmax >> vmin,
            # the value will be close to 1; if vmax = vmin + delta, the value will be less than 1. We expect the network can focus on the right position,
            # which implicitly wants vmax and vmin to be distinguishable.
            if (vmax) > 0:  # vmin, vmax >= 0 since softmax applied
                attention_map_resized = (attention_map_resized - vmin) / (vmax)
            attention_maps_j[:, :] = attention_map_resized[:, :]
            # =======================

            # plt.imshow(image_i)
            # plt.imshow(attention_maps_j, alpha=0.5)
            if is_show:
                # plt.axis('off')  # turn off axis and this command should be placed behind imshow() and before show()
                plt.show()
            if is_save:
                if valid_kp_mask[i, j] == 1:
                    # plt.savefig(save_image_root + '/visible_predicts/episode{}_img{}_{}{}.jpg'.format(episode_num, i, j, support_kp_categories[j]), bbox_inches='tight')
                    # RGB to BGR
                    show_cam_on_image(image_i[:, :, ::-1], attention_maps_j, save_image_root + '/visible_predicts/episode{}_img{}_{}{}.jpg'.format(episode_num, i, j, support_kp_categories[j]), mode='color')  # 'color' or 'gray'
                else:
                    # plt.savefig(save_image_root + '/invisible_predicts/episode{}_img{}_{}{}.jpg'.format(episode_num, i, j, support_kp_categories[j]), bbox_inches='tight')
                    # RGB to BGR
                    show_cam_on_image(image_i[:, :, ::-1], attention_maps_j, save_image_root + '/invisible_predicts/episode{}_img{}_{}{}.jpg'.format(episode_num, i, j, support_kp_categories[j]), mode='color')  # 'color' or 'gray'

    # plt.ioff()

def show_save_distinctive_maps(distinctive_maps : torch.Tensor, normalized_images : torch.Tensor, heatmap_process_method=3, episode_num=0, is_show=False, is_save=True, save_image_root='./episode_images/distinctive_maps', delete_old=False):
    (B, C, H, W) = normalized_images.shape
    (_, _, H2, W2) = distinctive_maps.shape  # B x 1 x H2 x W2, B maps
    scale = H / H2
    distinctive_maps2 = distinctive_maps.cpu().detach().numpy().copy().reshape(B, H2, W2)  # B x H2 x W2, B maps
    normalized_images2 = normalized_images.cpu().detach().numpy().transpose(0, 2, 3, 1)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if os.path.exists(save_image_root) == False:
        os.mkdir(save_image_root)
    if delete_old == True:
        for each_file in os.listdir(save_image_root):
            os.remove(os.path.join(save_image_root, each_file))

    # plt.ion()
    # plt.figure()
    for i in range(B):
        image_i = np.zeros((H, W, 3), dtype=np.float32)
        distinctive_map_j = np.zeros((H, W), dtype=np.float32)
        for c in range(3):
            image_i[:, :, c] = (normalized_images2[i, :, :, c] * std[c] + mean[c])
        distinctive_map_resized = cv2.resize(distinctive_maps2[i, :, :], (W, H), interpolation=cv2.INTER_CUBIC)
        # distinctive_map_resized = cv2.resize(distinctive_maps2[i, :, :], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if heatmap_process_method == 1:
            # =======================
            # normalization method 1
            # rescale the attention map according to its min value and max value
            vmin, vmax = np.min(distinctive_map_resized), np.max(distinctive_map_resized)
            if (vmax - vmin) > 0:  # vmin, vmax >= 0 since softmax applied
                distinctive_map_resized = (distinctive_map_resized - vmin) / (vmax - vmin)
            distinctive_map_j[:, :] = distinctive_map_resized[:, :]
            # =======================
        elif heatmap_process_method == 2:
            # rescale the attention map according to its min value and max value
            vmin, vmax = np.percentile(distinctive_map_resized, 0.5), np.percentile(distinctive_map_resized, 99.5)
            # vmin2, vmax2 = np.min(distinctive_map_resized), np.max(distinctive_map_resized)
            if (vmax - vmin) > 0:  # vmin, vmax >= 0 since softmax applied
                distinctive_map_resized = (distinctive_map_resized - vmin) / (vmax - vmin)
                distinctive_map_resized = distinctive_map_resized.clip(0, 1)
            distinctive_map_j[:, :] = distinctive_map_resized[:, :]
        elif heatmap_process_method == 3:
            # normalization method 2, advantage: it can reflect the difference between vmin and vmax. For example, if vmax >> vmin,
            # the value will be close to 1; if vmax = vmin + delta, the value will be less than 1. We expect the network can focus on the right position,
            # which implicitly wants vmax and vmin to be distinguishable.
            # rescale the attention map according to its min value and max value
            vmin, vmax = np.min(distinctive_map_resized), np.max(distinctive_map_resized)
            if (vmax) > 0:  # vmin, vmax >= 0 since softmax applied
                distinctive_map_resized = (distinctive_map_resized - vmin) / (vmax)
            distinctive_map_j[:, :] = distinctive_map_resized[:, :]
            # =======================

        # plt.imshow(image_i)
        # plt.imshow(distinctive_map_j, alpha=0.5)
        if is_show:
            # plt.axis('off')  # turn off axis and this command should be placed behind imshow() and before show()
            plt.show()
        if is_save:
            # plt.savefig(save_image_root + '/episode{}_img{}.jpg'.format(episode_num, i), bbox_inches='tight')
            # RGB to BGR
            show_cam_on_image(image_i[:, :, ::-1], distinctive_map_j, save_image_root + '/episode{}_img{}.jpg'.format(episode_num, i), mode='color')  # 'color' or 'gray'

    # plt.ioff()

def recover_kps(kps, current_image_length, scale_trans):
    '''
    :param kps: B x M x 2
    :param current_image_length: 368
    :param scale_trans: B x 6, (scale, xoffset, yoffset, bbx_area, pad_xoffset, pad_yoffset)
    :return:
    '''
    B = kps.shape[0]
    kps = kps / 2 + 0.5 # 0~1, since our kp's range is -1~1 thus it needs to perform x/2 + 0.5
    kps *= current_image_length - 1
    kps += (scale_trans[:, 1:3]).view(B, 1, 2)
    kps /= (scale_trans[:, 0]).view(B, 1, 1)
    # for batch_i in range(B):
        # single_label = kp_labels[batch_i, :, :]
        # single_label = single_label / 2 + 0.5  # 0~1
        # single_label *= square_image_length - 1
        # single_label += scale_trans[batch_i, 1:3].repeat(N, 1)
        # single_label /= scale_trans[batch_i, 0]
        # single_label = single_label.long()
    return kps