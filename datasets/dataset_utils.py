import os
import torch
import json
import argparse
from PIL import Image
import cv2
import numpy as np
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
# import sys
# sys.path.append('..')

CocoColors = [[0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0],
              [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],[0, 255, 85],
              [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [255, 255, 255], [0, 0, 0]]

def random_color():
    # generate random color where discrete interval is 32.
    levels = range(32, 256, 32)  # start, end, step
    color = tuple(random.choice(levels) for _ in range(3))
    return color

def filter_keypoints(keypoints_dict: dict):
    '''
    keypoints_dict: {'kp_type': [x, y, is_visible], ...}
    '''
    keypoints_filtered = OrderedDict()
    for kp_type in keypoints_dict.keys():
        if keypoints_dict[kp_type][2] == 0:
            continue
        keypoints_filtered[kp_type] = keypoints_dict[kp_type]

    return keypoints_filtered

def draw_skeletons(npimg, skeletons, all_keypoint_types, marker='circle', circle_radius=4, alpha=0.7, thickness=-1, stick_width=4, draw_keypoints=True, limbs=[], beta=0.6):
    '''
    :param npimg: bgr image, h x w x 3
    :param skeletons: list, a list of keypoint dictionaries, each dict represents a skeleton, like [{'kp_type1': [x, y, visible], ...}, ...]
    :param all_keypoint_types: a list of keypoint types, like ['kp_type1', 'kp_type2', ...], must to be given
    :param limbs: a list of pairs of keypoint ids, like [[kp_id1, kp_id2], [kp_id1, kp_id3], ...], each pair specifies a limb;
           if it is empty list, no limbs will be drawn
    '''
    image_h, image_w = npimg.shape[:2]  # npimg should be bgr image which has size of h x w x 3
    KEYPOINT_TYPES = all_keypoint_types # get_keypoints()
    KEYPOINT_IDS = list(range(len(KEYPOINT_TYPES)))
    PAIRS = limbs
    draw_limbs = True if len(limbs) > 0 else False
    for each_skeleton in skeletons:  # humans is [OrderedDict1, OrderDict2, ...]
        if draw_keypoints == True:
            # draw point
            # circle_radius = 4
            npimg_cur = np.copy(npimg)
            for i in KEYPOINT_IDS:
                if KEYPOINT_TYPES[i] not in each_skeleton.keys():
                    continue
                kp_type = KEYPOINT_TYPES[i]
                body_part = each_skeleton[kp_type]
                center = (int(body_part[0]), int(body_part[1]))
                if marker == 'circle':
                    # marker 1: circle, default size 4
                    # if thickness > 0:
                    #     cv2.circle(npimg_cur, center, circle_radius, [255, 255, 255], thickness=thickness+3)
                    cv2.circle(npimg_cur, center, circle_radius, CocoColors[i % len(CocoColors)], thickness=thickness)  # circle_thickness = -1, fill; otherwise does not fill
                    # if thickness < 0:
                    #     cv2.circle(npimg_cur, center, circle_radius, [255, 255, 255], thickness=2)
                elif marker == 'tilted_cross':
                    # marker 2: tilted cross, default thickness 2, size 6
                    cv2.drawMarker(npimg_cur, center, CocoColors[i % len(CocoColors)], cv2.MARKER_TILTED_CROSS, thickness=2, markerSize=circle_radius)
                elif marker == 'cross':
                    # marker 3: cross, default thickness 2, size 8
                    cv2.drawMarker(npimg_cur, center, CocoColors[i % len(CocoColors)], cv2.MARKER_CROSS, thickness=2, markerSize=circle_radius)
                elif marker == 'diamond':
                    # marker 4: diamond, default thickness 2, size 8
                    cv2.drawMarker(npimg_cur, center, CocoColors[i % len(CocoColors)], markerType=cv2.MARKER_DIAMOND, thickness=2, markerSize=circle_radius)
                elif marker == 'triangle_up':
                    # marker 5: triangle up, default thickness 2, size 6
                    cv2.drawMarker(npimg_cur, center, CocoColors[i % len(CocoColors)], markerType=cv2.MARKER_TRIANGLE_UP, thickness=2, markerSize=circle_radius)
                elif marker == 'square':
                    # marker 6: square, default thickness 2, size 6
                    cv2.drawMarker(npimg_cur, center, CocoColors[i % len(CocoColors)], markerType=cv2.MARKER_SQUARE, thickness=2, markerSize=circle_radius)

            # npimg = cv2.addWeighted(npimg, 0.3, npimg_cur, 0.7, 0)
            npimg = cv2.addWeighted(npimg, 1-alpha, npimg_cur, alpha, 0)

        if draw_limbs == True:
            # draw line
            # stick_width = 4
            npimg_cur = np.copy(npimg)
            for pair_order, pair_ids in enumerate(PAIRS):
                if KEYPOINT_TYPES[pair_ids[0]] not in each_skeleton.keys() or KEYPOINT_TYPES[pair_ids[1]] not in each_skeleton.keys():
                    continue
                # cv2.line(npimg_cur, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 2)
                kp_type1, kp_type2 = KEYPOINT_TYPES[pair_ids[0]], KEYPOINT_TYPES[pair_ids[1]]
                X = [each_skeleton[kp_type1][0], each_skeleton[kp_type2][0]]
                Y = [each_skeleton[kp_type1][1], each_skeleton[kp_type2][1]]
                meanX = np.mean(X)
                meanY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1])) # atan(x/y)
                polygon = cv2.ellipse2Poly((int(meanX), int(meanY)), (int(length / 2), int(stick_width / 2)), int(angle), 0, 360, 1)

                cv2.fillConvexPoly(npimg_cur, polygon, CocoColors[pair_order % len(CocoColors)])
                # npimg = cv2.addWeighted(npimg, 0.4, npimg_cur, 0.6, 0)
                npimg = cv2.addWeighted(npimg, 1-beta, npimg_cur, beta, 0)

    return npimg

# draw keypoints from path
def draw_instance(image_path, keypoints, full_keypoint_types, limbs=[], visible_bounds=None, hightlight_keypoint_types=None, save_root=None, save_prefix="", is_show=True, circle_radius=4, stick_width=4):
    '''
    :param image_path:
    :param keypoints: dictionary of keypoints, {'l_eye':[x1,y1,is_visible], 'r_eye':[x2,y2,is_visible], ...}
    :param draw_limbs: True or False
    :param all_keypoint_types: a list of keypoint types, like ['kp_type1', 'kp_type2', ...], must to be given
    :param limbs: a list of pairs of keypoint ids, like [[kp_id1, kp_id2], [kp_id1, kp_id3], ...], each pair specifies a limb;
           if it is empty list, no limbs will be drawn
    :param visible_bounds: [h, w, xmin, ymin]
    :param hightlight_keypoint_types: ['l_eye', 'r_eye', ...]
    :param save_root:
    :param is_show
    :return:
    '''
    image = cv2.imread(image_path)
    KEYPOINT_TYPES = full_keypoint_types
    KEYPOINT_TYPE_IDS = OrderedDict(zip(KEYPOINT_TYPES, range(len(KEYPOINT_TYPES))))

    keypoints_filtered = OrderedDict()
    for kp_type in keypoints.keys():
        if keypoints[kp_type][2] == 0:
            continue
        keypoints_filtered[kp_type] = keypoints[kp_type]

    # image_out = draw_skeletons(image, [keypoints_filtered], KEYPOINT_TYPES, circle_radius=4, stick_width=4, draw_keypoints=True, limbs=limbs)
    image_out = draw_skeletons(image, [keypoints_filtered], KEYPOINT_TYPES, circle_radius=circle_radius, stick_width=stick_width, draw_keypoints=True, limbs=limbs)
    if visible_bounds != None:
        cv2.rectangle(image_out, (int(visible_bounds[2]), int(visible_bounds[3])),
                      (int(visible_bounds[2] + visible_bounds[1]), int(visible_bounds[3] + visible_bounds[0])), (0, 255, 0))

    if hightlight_keypoint_types != None:
        image_curr = np.copy(image_out)
        for kp_type in hightlight_keypoint_types:
            if kp_type not in keypoints.keys() or keypoints[kp_type][2] == 0:
                continue
            body_part = keypoints[kp_type]
            center = (int(body_part[0]), int(body_part[1]))
            cv2.circle(image_curr, center, 15, CocoColors[KEYPOINT_TYPE_IDS[kp_type] % len(CocoColors)], thickness=-1)  # 15
        image_out = cv2.addWeighted(image_out, 0.4, image_curr, 0.6, 0)

    if is_show:
        plt.imshow(image_out[:, :, ::-1])
        plt.show()

    if save_root != None:
        _, filename = os.path.split(image_path)
        filename_with_ext, ext = os.path.splitext(filename)
        cv2.imwrite(os.path.join(save_root, save_prefix + "_" + filename_with_ext + ext), image_out)

    return image_out

def draw_markers(image, keypoint_dict, marker='circle', color=[255,255,255], circle_radius=10, thickness=-1, alpha=1.0):
    npimg_cur = np.copy(image)
    for i, kp_type in enumerate(keypoint_dict.keys()):
        body_part = keypoint_dict[kp_type]
        if len(body_part)==3 and body_part[2] == -1:  # (x, y, v), not visible
            continue
        center = (int(body_part[0]), int(body_part[1]))
        if marker == 'circle':
            # marker 1: circle, default size 4
            # if thickness > 0:
            #     cv2.circle(npimg_cur, center, circle_radius, [255, 255, 255], thickness=thickness+3)
            cv2.circle(npimg_cur, center, circle_radius, color, thickness=thickness)  # circle_thickness = -1, fill; otherwise does not fill
            cv2.putText(npimg_cur, str(i), center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), thickness=1)
            # if thickness < 0:
            #     cv2.circle(npimg_cur, center, circle_radius, [255, 255, 255], thickness=2)
        elif marker == 'tilted_cross':
            # marker 2: tilted cross, default thickness 2, size 6
            cv2.drawMarker(npimg_cur, center, color, cv2.MARKER_TILTED_CROSS, thickness=thickness, markerSize=circle_radius)
        elif marker == 'cross':
            # marker 3: cross, default thickness 2, size 8
            cv2.drawMarker(npimg_cur, center, color, cv2.MARKER_CROSS, thickness=2, markerSize=circle_radius)
        elif marker == 'diamond':
            # marker 4: diamond, default thickness 2, size 8
            cv2.drawMarker(npimg_cur, center, color, markerType=cv2.MARKER_DIAMOND, thickness=2,
                           markerSize=circle_radius)
        elif marker == 'triangle_up':
            # marker 5: triangle up, default thickness 2, size 6
            cv2.drawMarker(npimg_cur, center, color, markerType=cv2.MARKER_TRIANGLE_UP, thickness=2,
                           markerSize=circle_radius)
        elif marker == 'square':
            # marker 6: square, default thickness 2, size 6
            cv2.drawMarker(npimg_cur, center, color, markerType=cv2.MARKER_SQUARE, thickness=2,
                           markerSize=circle_radius)

    # npimg = cv2.addWeighted(npimg, 0.3, npimg_cur, 0.7, 0)
    npimg = cv2.addWeighted(image, 1 - alpha, npimg_cur, alpha, 0)

    return npimg