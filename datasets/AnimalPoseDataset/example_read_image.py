import copy
import os
import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import math
import json
from PIL import Image

from datasets.dataset_utils import CocoColors, draw_skeletons, draw_instance


# 20 keypoints for animal pose dataset
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'l_eye',
        'r_eye',
        'l_ear',
        'r_ear',
        'nose',
        'throat',
        'withers',
        'tail',
        'l_f_leg',
        'r_f_leg',
        'l_b_leg',
        'r_b_leg',
        'l_f_knee',
        'r_f_knee',
        'l_b_knee',
        'r_b_knee',
        'l_f_paw',
        'r_f_paw',
        'l_b_paw',
        'r_b_paw'
    ]
    return keypoints

# 20 defined limbs
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('nose'), keypoints.index('l_eye')],
        [keypoints.index('nose'), keypoints.index('r_eye')],
        [keypoints.index('l_eye'), keypoints.index('l_ear')],
        [keypoints.index('r_eye'), keypoints.index('r_ear')],
        [keypoints.index('withers'), keypoints.index('tail')],
        [keypoints.index('withers'), keypoints.index('l_f_leg')],
        [keypoints.index('l_f_leg'), keypoints.index('l_f_knee')],
        [keypoints.index('l_f_knee'), keypoints.index('l_f_paw')],
        [keypoints.index('withers'), keypoints.index('r_f_leg')],
        [keypoints.index('r_f_leg'), keypoints.index('r_f_knee')],
        [keypoints.index('r_f_knee'), keypoints.index('r_f_paw')],
        [keypoints.index('tail'), keypoints.index('l_b_leg')],
        [keypoints.index('l_b_leg'), keypoints.index('l_b_knee')],
        [keypoints.index('l_b_knee'), keypoints.index('l_b_paw')],
        [keypoints.index('tail'), keypoints.index('r_b_leg')],
        [keypoints.index('r_b_leg'), keypoints.index('r_b_knee')],
        [keypoints.index('r_b_knee'), keypoints.index('r_b_paw')],
        # [keypoints.index('withers'), keypoints.index('l_ear')],
        # [keypoints.index('withers'), keypoints.index('r_ear')],
        # [keypoints.index('withers'), keypoints.index('nose')],
        [keypoints.index('throat'), keypoints.index('withers')],
        # [keypoints.index('throat'), keypoints.index('l_eye')],
        # [keypoints.index('throat'), keypoints.index('r_eye')]
        [keypoints.index('throat'), keypoints.index('nose')]
    ]
    return kp_lines

# read one xml file and return a sample which contains
# image filename,
# category,
# N keypoints, each of which has 1 x 3 elements, representing [x, y, visible(0 or 1)]
# visible bounds, [height, width, xmin, ymin]
def readxml(path, dataset_type='animalpose_anno2'):
    tree = ET.parse(path)
    root = tree.getroot() # node: annotation

    # find xml node, we can use find() function
    # as for get 'tag', 'attribute', or 'text' from one node, we can directly get them
    filename = str(root.find('image').text)
    category = str(root.find('category').text)

    node_visible_bounds = root.find('visible_bounds')
    if dataset_type == 'animalpose_anno2':
        visible_bounds = [int(node_visible_bounds.attrib['height']),
                          int(node_visible_bounds.attrib['width']),
                          int(node_visible_bounds.attrib['xmin']),
                          int(node_visible_bounds.attrib['xmax'])
                          ]
    elif dataset_type =='PASCAL2011_animal_annotation':
        visible_bounds = [int(float(node_visible_bounds.attrib['height'])),
                          int(float(node_visible_bounds.attrib['width'])),
                          int(float(node_visible_bounds.attrib['xmin'])),
                          int(float(node_visible_bounds.attrib['ymin']))
                          ]

    node_keypoints = root.find('keypoints')
    KEYPOINT_TYPES = get_keypoints()
    # initialize the ordered keypoint names
    keypoints = OrderedDict(zip(KEYPOINT_TYPES, ['']*len(KEYPOINT_TYPES)))
    for node_kp in node_keypoints.iter('keypoint'):
        joint_type = str(node_kp.attrib['name'])
        x = int(float(node_kp.attrib['x']))
        y = int(float(node_kp.attrib['y']))
        is_visible = int(node_kp.attrib['visible'])
        kp = [x, y, is_visible]
        if dataset_type == 'animalpose_anno2':
            if joint_type == 'L_eye':
                keypoints['l_eye'] = kp
            elif joint_type == 'R_eye':
                keypoints['r_eye'] = kp
            elif joint_type == 'L_ear':
                keypoints['l_ear'] = kp
            elif joint_type == 'R_ear':
                keypoints['r_ear'] = kp
            elif joint_type == 'Nose':
                keypoints['nose'] = kp
            elif joint_type == 'Throat':
                keypoints['throat'] = kp
            elif joint_type == 'Tail':
                keypoints['tail'] = kp
            elif joint_type == 'withers':
                keypoints['withers'] = kp
            elif joint_type == 'L_F_elbow':
                keypoints['l_f_leg'] = kp
            elif joint_type == 'R_F_elbow':
                keypoints['r_f_leg'] = kp
            elif joint_type == 'L_B_elbow':
                keypoints['l_b_leg'] = kp
            elif joint_type == 'R_B_elbow':
                keypoints['r_b_leg'] = kp
            elif joint_type == 'L_F_knee':
                keypoints['l_f_knee'] = kp
            elif joint_type == 'R_F_knee':
                keypoints['r_f_knee'] = kp
            elif joint_type == 'L_B_knee':
                keypoints['l_b_knee'] = kp
            elif joint_type == 'R_B_knee':
                keypoints['r_b_knee'] = kp
            elif joint_type == 'L_F_paw':
                keypoints['l_f_paw'] = kp
            elif joint_type == 'R_F_paw':
                keypoints['r_f_paw'] = kp
            elif joint_type == 'L_B_paw':
                keypoints['l_b_paw'] = kp
            elif joint_type == 'R_B_paw':
                keypoints['r_b_paw'] = kp
        elif dataset_type =='PASCAL2011_animal_annotation':
            if joint_type == 'L_Eye':
                keypoints['l_eye'] = kp
            elif joint_type == 'R_Eye':
                keypoints['r_eye'] = kp
            elif joint_type == 'L_EarBase':
                keypoints['l_ear'] = kp
            elif joint_type == 'R_EarBase':
                keypoints['r_ear'] = kp
            elif joint_type == 'Nose':
                keypoints['nose'] = kp
            elif joint_type == 'Throat':
                keypoints['throat'] = kp
            elif joint_type == 'TailBase':
                keypoints['tail'] = kp
            elif joint_type == 'Withers':
                keypoints['withers'] = kp
            elif joint_type == 'L_F_Elbow':
                keypoints['l_f_leg'] = kp
            elif joint_type == 'R_F_Elbow':
                keypoints['r_f_leg'] = kp
            elif joint_type == 'L_B_Elbow':
                keypoints['l_b_leg'] = kp
            elif joint_type == 'R_B_Elbow':
                keypoints['r_b_leg'] = kp
            elif joint_type == 'L_F_Knee':
                keypoints['l_f_knee'] = kp
            elif joint_type == 'R_F_Knee':
                keypoints['r_f_knee'] = kp
            elif joint_type == 'L_B_Knee':
                keypoints['l_b_knee'] = kp
            elif joint_type == 'R_B_Knee':
                keypoints['r_b_knee'] = kp
            elif joint_type == 'L_F_Paw':
                keypoints['l_f_paw'] = kp
            elif joint_type == 'R_F_Paw':
                keypoints['r_f_paw'] = kp
            elif joint_type == 'L_B_Paw':
                keypoints['l_b_paw'] = kp
            elif joint_type == 'R_B_Paw':
                keypoints['r_b_paw'] = kp

    return filename, category, keypoints, visible_bounds

def reformat_gt():
    AnimalPose_json_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt'
    cat_anno_path = AnimalPose_json_root + '/cat.json'
    dog_anno_path = AnimalPose_json_root + '/dog.json'
    cow_anno_path = AnimalPose_json_root + '/cow.json'
    horse_anno_path = AnimalPose_json_root + '/horse.json'
    sheep_anno_path = AnimalPose_json_root + '/sheep.json'
    path = [cat_anno_path, dog_anno_path, cow_anno_path, horse_anno_path, sheep_anno_path]
    for i in range(len(path)):
        with open(path[i], 'r') as fin:
            dataset = json.load(fin)
            fin.close()
        dataset2 = copy.deepcopy(dataset)
        KEYPOINT_TYPES = get_keypoints()
        # modify keypoint dict into keypoint list
        # for j in range(len(dataset2)):
        #     keypoints = []
        #     for kp_type in KEYPOINT_TYPES:
        #         keypoints.append(dataset2[j]['keypoints'][kp_type])
        #     dataset2[j]['keypoints'] = keypoints
        # modify name
        for j in range(len(dataset2)):
            instance = {}
            instance['filename'] = dataset2[j]['filename']
            instance['category'] = dataset2[j]['category']
            instance['keypoints'] = dataset2[j]['keypoints']
            instance['bbx'] = dataset2[j]['visible_bounds']
            instance['w_h'] = dataset2[j]['w_h']
            dataset2[j] = instance
        with open(path[i], 'w') as fout:
            json.dump(dataset2, fout)
            fout.close()
    print('ok')

def change_xml_to_json(anno_folder_path, output_folder_path):
    '''
    change all the xml labels in one folder to json file

    :input folder_path:
    :return:
    the format of stored annotation is like in the following,
    [sample1, sample2, ...]
    each sample is a dictionary, which is like
    {'filename': 'co1.jpeg',
    'category': 'cow',
    'keypoints': OrderedDict([('l_eye', [135, 122, 1]), ('r_eye', [0, 0, 0]), ('l_ear', [200, 83, 1]), ('r_ear', [50, 106, 1]), ('nose', [54, 221, 1]), ('throat', [108, 259, 1]), ('withers', [0, 0, 0]), ('tail', [0, 0, 0]), ('l_f_leg', [0, 0, 0]), ('r_f_leg', [0, 0, 0]), ('l_b_leg', [0, 0, 0]), ('r_b_leg', [0, 0, 0]), ('l_f_knee', [0, 0, 0]), ('r_f_knee', [0, 0, 0]), ('l_b_knee', [0, 0, 0]), ('r_b_knee', [0, 0, 0]), ('l_f_paw', [0, 0, 0]), ('r_f_paw', [0, 0, 0]), ('l_b_paw', [0, 0, 0]), ('r_b_paw', [0, 0, 0])]),
    'visible_bounds': [256, 207, 0, 42],
    }
    each 3-tuple keypoint represents [x, y, isvisible]
    each rectangle bound represents [height, width, xmin, ymin]
    '''
    folders = os.listdir(anno_folder_path)
    for each_folder in folders:
        print(each_folder)
        class_root = os.path.join(anno_folder_path, each_folder)  # root for each class, like 'cat', 'dog', ...
        samples_per_class = []
        for each_xml in os.listdir(class_root):
            xml_path = os.path.join(class_root, each_xml)
            # filename, category, keypoints, visible_bounds = readxml(xml_path)
            filename, category, keypoints, visible_bounds = readxml(xml_path, 'PASCAL2011_animal_annotation')
            keypoints = [keypoints[kp_type] for kp_type in get_keypoints()]  # only use its coordinates
            # sample = {'filename': filename, 'category': category, 'keypoints': keypoints, 'bbx': visible_bounds}
            sample = {'filename':filename+'.jpg', 'category':category, 'keypoints':keypoints, 'bbx':visible_bounds}
            samples_per_class.append(sample)

        with open(os.path.join(output_folder_path, each_folder+'.json'), 'w') as fout:
            json.dump(samples_per_class, fout)
            fout.close()



def main1():
    # only can run one change operation per time due to format i
    # change_xml_to_json('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_anno2',
    #                    '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_anno2_json')
    # change_xml_to_json('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation',
    #                  '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation_json')
    # exit()

    keypoint_types = get_keypoints()
    # OrderedDict([('l_eye', 0), ('r_eye', 1), ('l_ear', 2), ('r_ear', 3), ('nose', 4), ('throat', 5), ('withers', 6),
    # ('tail', 7), ('l_f_leg', 8), ('r_f_leg', 9), ('l_b_leg', 10), ('r_b_leg', 11), ('l_f_knee', 12), ('r_f_knee', 13),
    # ('l_b_knee', 14), ('r_b_knee', 15), ('l_f_paw', 16), ('r_f_paw', 17), ('l_b_paw', 18), ('r_b_paw', 19)])
    keypoint_ids = OrderedDict(zip(keypoint_types, range(len(keypoint_types))))
    print(keypoint_ids)

    # label-reading method 1
    # filename, category, keypoints, visible_bounds = readxml('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_anno2/cat/ca23.xml')
    # filename, category, keypoints, visible_bounds = readxml('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation/cat/2007_000528_1.xml', \
    #                                                        dataset_type='PASCAL2011_animal_annotation')

    anno_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation_json'
    anno_root2 = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt'
    for files in os.listdir(anno_root2):
        print(files)
        # label-reading method 2
        samples = []
        # json_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_anno2_json/' + files
        # json_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation_json/' + files
        json_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt/' + files

        with open(json_path, 'r') as fin:
            samples = json.load(fin)
            fin.close()

        for one_sample in samples:
            # one_sample = samples[0]  # choose one sample
            filename = one_sample['filename']
            category = one_sample['category']
            keypoints = one_sample['keypoints']
            visible_bounds= one_sample['visible_bounds']
            # if (filename == 'sh1.jpg'):
            #     print('here')
            print(filename)
            # print(filename, category, keypoints, visible_bounds)

            AnimalPose_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_image_part2'
            VOC_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL-VOC2010-TrainVal/VOCdevkit/VOC2011/JPEGImages'
            combined_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images'

            # image_path = os.path.join(AnimalPose_image_root, category, filename)
            # image_path = os.path.join(AnimalPose_image_root, filename)
            # image_path = os.path.join(VOC_image_root, filename + '.jpg')
            # image_path = os.path.join(VOC_image_root, filename)
            image_path = os.path.join(combined_image_root, category, filename)
            image = cv2.imread(image_path)

            keypoints_filtered = OrderedDict()
            for kp_type in keypoints.keys():
                if keypoints[kp_type][2] == 0:
                    continue
                keypoints_filtered[kp_type] = keypoints[kp_type]


            image_out = draw_skeletons(image, [keypoints_filtered], keypoint_types)
            cv2.rectangle(image_out, (visible_bounds[2], visible_bounds[3]), (visible_bounds[2]+visible_bounds[1], visible_bounds[3]+visible_bounds[0]),(0, 255, 0))
            # plt.imshow(image_out[:,:,::-1])
            # plt.show()
            # cv2.imwrite('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/'+filename+'_horse.jpg', image_out)

            name,_ = os.path.splitext(files)
            # save_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/labelsshow/voc2011/' + name
            save_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/labelshow2/' + name
            if os.path.exists(save_root) == False:
                os.mkdir(save_root)
            cv2.imwrite(save_root + '/' + filename, image_out)

def main2():
    print('main2')
    json_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation_json/' + 'cat.json'
    with open(json_path, 'r') as fin:
        samples = json.load(fin)
        fin.close()

    one_sample = samples[0]  # choose one sample
    filename = one_sample['filename']
    category = one_sample['category']
    keypoints = one_sample['keypoints']
    visible_bounds = one_sample['visible_bounds']
    # if (filename == 'sh1.jpg'):
    #     print('here')
    print(filename)
    # print(filename, category, keypoints, visible_bounds)

    AnimalPose_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_image_part2'
    VOC_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL-VOC2010-TrainVal/VOCdevkit/VOC2011/JPEGImages'

    # image_path = os.path.join(AnimalPose_image_root, category, filename)
    # image_path = os.path.join(AnimalPose_image_root, filename)
    # image_path = os.path.join(VOC_image_root, filename + '.jpg')
    image_path = os.path.join(VOC_image_root, filename)
    image = cv2.imread(image_path)

    keypoints_filtered = OrderedDict()
    for kp_type in keypoints.keys():
        if keypoints[kp_type][2] == 0:
            continue
        keypoints_filtered[kp_type] = keypoints[kp_type]

    KEYPOINT_TYPES = get_keypoints()
    image_out = draw_skeletons(image, [keypoints_filtered], KEYPOINT_TYPES, draw_keypoints=True, draw_limbs=False)
    cv2.rectangle(image_out, (visible_bounds[2], visible_bounds[3]),
                  (visible_bounds[2] + visible_bounds[1], visible_bounds[3] + visible_bounds[0]), (0, 255, 0))
    plt.imshow(image_out[:,:,::-1])
    plt.show()
    # cv2.imwrite('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/'+filename+'_horse.jpg', image_out)

def main3():
    print('main3')
    # label-reading method 1
    # filename, category, keypoints, visible_bounds = readxml('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_anno2/cat/ca23.xml')
    filename, category, keypoints, visible_bounds = readxml('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation/cow/2008_002278_4.xml', \
                                                           dataset_type='PASCAL2011_animal_annotation')
    # if (filename == 'sh1.jpg'):
    #     print('here')
    print(filename)
    # print(filename, category, keypoints, visible_bounds)

    AnimalPose_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_image_part2'
    VOC_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL-VOC2010-TrainVal/VOCdevkit/VOC2011/JPEGImages'

    # image_path = os.path.join(AnimalPose_image_root, category, filename)
    # image_path = os.path.join(AnimalPose_image_root, filename)
    image_path = os.path.join(VOC_image_root, filename + '.jpg')
    # image_path = os.path.join(VOC_image_root, filename)
    image = cv2.imread(image_path)

    keypoints_filtered = OrderedDict()
    for kp_type in keypoints.keys():
        if keypoints[kp_type][2] == 0:
            continue
        keypoints_filtered[kp_type] = keypoints[kp_type]

    KEYPOINT_TYPES = get_keypoints()
    image_out = draw_skeletons(image, [keypoints_filtered], KEYPOINT_TYPES)
    cv2.rectangle(image_out, (visible_bounds[2], visible_bounds[3]),
                  (visible_bounds[2] + visible_bounds[1], visible_bounds[3] + visible_bounds[0]), (0, 255, 0))
    plt.imshow(image_out[:, :, ::-1])
    plt.show()
    # cv2.imwrite('/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/'+filename+'_horse.jpg', image_out)

if __name__=='__main__':
    # reformat_gt()
    # main1()
    # exit(0)

    # json_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation_json/' + 'horse.json'
    json_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt' + '/cat.json'
    with open(json_path, 'r') as fin:
        samples = json.load(fin)
        fin.close()

    one_sample = samples[2]  # choose one sample
    filename = one_sample['filename']
    category = one_sample['category']
    keypoints = one_sample['keypoints']
    visible_bounds = one_sample['bbx']
    image_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images/' + category + '/' + filename
    KEYPOINT_TYPES = get_keypoints()
    keypoints_dict = OrderedDict(zip(KEYPOINT_TYPES, keypoints))
    LIMBS = kp_connections(KEYPOINT_TYPES)
    draw_instance(image_path, keypoints_dict, KEYPOINT_TYPES, limbs=LIMBS, visible_bounds=visible_bounds, hightlight_keypoint_types=None, is_show=True)
    # if (filename == 'sh1.jpg'):
    #     print('here')
    print(filename)
    print(one_sample)
    # print(filename, category, keypoints, visible_bounds)

    AnimalPose_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/animalpose_image_part2'
    VOC_image_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL-VOC2010-TrainVal/VOCdevkit/VOC2011/JPEGImages'

    # image_path = os.path.join(AnimalPose_image_root, category, filename)
    # image_path = os.path.join(AnimalPose_image_root, filename)
    # image_path = os.path.join(VOC_image_root, filename + '.jpg')
    image_path = os.path.join(VOC_image_root, filename)

    npimg = draw_instance(image_path, keypoints_dict, KEYPOINT_TYPES, limbs=LIMBS, hightlight_keypoint_types=['l_eye', 'r_eye', 'nose'], is_show=False)
    pt1 = keypoints[KEYPOINT_TYPES.index('r_f_leg')]
    pt2 = keypoints[KEYPOINT_TYPES.index('r_f_knee')]

    npimg_cur = np.copy(npimg)

    stick_width = 8
    # cv2.line(npimg_cur, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 2)
    X = [pt1[0], pt2[0]]
    Y = [pt1[1], pt2[1]]
    meanX = np.mean(X)
    meanY = np.mean(Y)
    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    stick_width = length // 4
    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1])) # atan(y/x)
    print(angle)
    polygon = cv2.ellipse2Poly((int(meanX), int(meanY)), (int(length / 2), int(stick_width / 2)), int(angle), 0, 360, 1)  # clockwise rotation (+), anti-clockwise (-)
    n = polygon.shape[0]
    polygon2 = polygon[0: int(n*2/4), :]
    cv2.fillConvexPoly(npimg_cur, polygon, CocoColors[3])
    npimg = cv2.addWeighted(npimg, 0.4, npimg_cur, 0.6, 0)
    plt.imshow(npimg[:, :, ::-1])
    plt.show()
