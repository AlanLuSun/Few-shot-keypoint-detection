import PIL
from PIL import Image
import numpy as np
import torchvision
import torch
import copy

from abc import ABCMeta, abstractmethod
from functools import partial, reduce
import numbers
import cv2
import random

import matplotlib.pyplot as plt

class Scale(object):
    def __init__(self, scale, interpolation=Image.BILINEAR):
        """zoom in or zoom out the image while not changing apsect ratio

        Arguments:
            scale {float} -- zoom in or zoom out ratio

        Keyword Arguments:
            interpolation {int} -- 插值方法 (default: {Image.BILINEAR})

        Examples:
            >>> scale = Scale(2)
            >>> image = Image.open('image.png')
            >>> image = scale(image)
        """
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, image):
        """apply scale transform to image

        Arguments:
            image {Image.Image} -- image waited to be scaled

        Returns:
            Image.Image -- image after scaled
        """
        return image.resize(
            (int(image.size[0] * self.scale), int(image.size[1] * self.scale)),
            resample=self.interpolation
        )


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        """将图像缩放至目标大小
        本对象用于兼容旧版本torchvision

        Arguments:
            scale {{int,...}} -- 目标大小 (width × height)

        Keyword Arguments:
            interpolation {int} -- 插值方法 (default: {Image.BILINEAR})

        Examples:
            >>> resize = Resize(size=(800, 600))
            >>> image = Image.open('image.png')
            >>> image = resize(image)
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        return image.resize(self.size, resample=self.interpolation)


class Preprocess(object):
    def __init__(self, square_image_length):
        self.square_image_length = square_image_length
        self.meta = None

    def __call__(self, pil_image, keypoint_annos):
        '''
        :param pil_image:
        :param keypoint_annos: N x 2 numpy arrays and each row represents a keypoint [x, y]
        :return:
        '''
        w, h = pil_image.size
        self.meta = {
            'scale': 1.0,
            'offset': np.array((0.0, 0.0)),
            'valid_area': np.array((0.0, 0.0, w, h)),
            'hflip': False,
            'width_height': np.array((w, h))
        }

        # undistorted resize along long side
        target_long = self.square_image_length
        if w < h:
            scale = target_long / h
            target_shape = (int(round(w * scale)), target_long)
        else:
            scale = target_long / w
            target_shape = (target_long, int(round(h * scale)))
        pil_image = pil_image.resize(target_shape, resample=PIL.Image.CUBIC)

        self.meta['scale'] = scale
        # self.meta['valid_area'][2:] = np.array(target_shape)

        # center padding
        w2, h2 = pil_image.size
        left = int((target_long - w2) / 2.0)
        top = int((target_long - h2) / 2.0)
        ltrb = (
            left,
            top,
            target_long - w2 - left,
            target_long - h2 - top,
        )
        # pad image
        pil_image = torchvision.transforms.functional.pad(
            pil_image, ltrb, fill=(124, 116, 104))

        self.meta['offset'] -= ltrb[:2]
        # self.meta['valid_area'][2:] += ltrb[:2]
        # self.meta['valid_area'][0:2] += ltrb[:2]

        # scale and pad annotations
        # P' = P * scale - offset, P = (P' + offset) / scale
        for keypoint in keypoint_annos:
            keypoint[:] = keypoint[:] * self.meta['scale'] - self.meta['offset']
        # P'' = P' / (square_image_length-1), P' = P'' * (square_image_length-1)
        # normalized to 0~1
        keypoint_annos = keypoint_annos / (self.square_image_length - 1)

        # scale, offset_x, offset_y
        scale_trans = np.array([self.meta['scale'], self.meta['offset'][0], self.meta['offset'][1]])
        return pil_image, keypoint_annos, scale_trans



class PreprocessAbstract(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image, annos, meta):
        pass


class Compose(PreprocessAbstract):
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list
        # print(self.__class__.__name__)

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)
        augmentations = [partial(aug_meth) for aug_meth in self.preprocess_list]
        image, anns, meta = reduce(
            lambda md_i_mm, f: f(*md_i_mm),
            augmentations,
            (image, anns, meta)
        )
        return image, anns, meta


class RandomApply(PreprocessAbstract):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.probability = p

    def __call__(self, image, anns, meta):
        # probability ranges 0~1 following uniform distribution
        if float(torch.rand(1).item()) > self.probability:
            return image, anns, meta
        return self.transform(image, anns, meta)

class RandomGrayscale(PreprocessAbstract):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, anns, meta):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if image.mode == 'L' else 3
        if random.random() < self.p:
            image = torchvision.transforms.functional.to_grayscale(image, num_output_channels=num_output_channels)
        return image, anns, meta


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ColorJitter(PreprocessAbstract):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: torchvision.transforms.functional.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, image, anns, meta):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(image), anns, meta

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RelativeResize(PreprocessAbstract):
    def __init__(self, scale_range=(0.75, 1.25), resample=PIL.Image.BILINEAR):
        self.scale_range = scale_range
        self.resample = resample

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        if isinstance(self.scale_range, tuple):
            scale_factor = (
                self.scale_range[0] +
                torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
            )
        else:
            scale_factor = self.scale_range

        # R, G, B, saliency_image2 = image.split()
        # image2 = Image.merge('RGB', (R, G, B))
        # image2.show()
        # saliency_image2.show()
        # image.mode = 'RGBa'

        # There is a bug for PIL.Image.Image.resize(): this function will modify the pixel value if the image mode is 'RGBA'.
        # Thus we manipulate the mode to avoid this bug. After applying resize(), the mode will be recover.
        manipulate_image_mode_to_handle_bug=False
        if image.mode == 'RGBA':
            image.mode = 'RGBa'
            manipulate_image_mode_to_handle_bug = True

        w, h = image.size  # image should be PIL image
        target_shape = (int(w * scale_factor), int(h * scale_factor))
        image = image.resize(target_shape, self.resample)

        if manipulate_image_mode_to_handle_bug == True:
            image.mode='RGBA'

        # image.mode = 'RGBa'
        # R, G, B, saliency_image2 = image.split()
        # image2 = Image.merge('RGB', (R, G, B))
        # image2.show()
        # saliency_image2.show()

        x_scale = target_shape[0] / w
        y_scale = target_shape[1] / h

        # modify annotations
        anns['keypoints'][:, 0] *= x_scale  # (anns['keypoints'][:, 0] + 0.5) * scale - 0.5  # N x 3, (x, y, is_visible) at each row
        anns['keypoints'][:, 1] *= y_scale  # (anns['keypoints'][:, 1] + 0.5) * scale - 0.5
        anns['bbox'][0] *= x_scale  # (xmin, ymin, w, h)
        anns['bbox'][1] *= y_scale
        anns['bbox'][2] *= x_scale
        anns['bbox'][3] *= y_scale

        # modify geometry transformation parameters
        meta['offset'] *= scale_factor  # (offset_x, offset_y)
        meta['scale'] *= scale_factor   # scale
        # meta['valid_area'][:2] *= scale_factor  # (xmin, ymin, w, h)
        # meta['valid_area'][2:] *= scale_factor

        return image, anns, meta


class Resize(PreprocessAbstract):
    def __init__(self, longer_length, resample=PIL.Image.BILINEAR):
        self.longer_length = longer_length
        self.resample = resample  # PIL.Image.BILINEAR, PIL.Image.CUBIC, PIL.Image.BICUBIC
        # print(self.__class__.__name__)

    def __call__(self, image, anns, meta):
        w, h = image.size  # image should be PIL image
        # undistorted resize along long side
        target_long = self.longer_length
        if w < h:
            scale = target_long / h
            target_shape = (int(round(w * scale)), target_long)
        else:
            scale = target_long / w
            target_shape = (target_long, int(round(h * scale)))

        # There is a bug for PIL.Image.Image.resize(): this function will modify the pixel value if the image mode is 'RGBA'.
        # Thus we manipulate the mode to avoid this bug. After applying resize(), the mode will be recover.
        manipulate_image_mode_to_handle_bug=False
        if image.mode == 'RGBA':
            image.mode = 'RGBa'
            manipulate_image_mode_to_handle_bug = True

        image = image.resize(target_shape, resample=self.resample)

        if manipulate_image_mode_to_handle_bug == True:
            image.mode='RGBA'

        x_scale = target_shape[0] / w
        y_scale = target_shape[1] / h

        # modify annotations
        anns['keypoints'][:, 0] *= x_scale  # (anns['keypoints'][:, 0] + 0.5) * scale - 0.5  # N x 3, (x, y, is_visible) at each row
        anns['keypoints'][:, 1] *= y_scale  # (anns['keypoints'][:, 1] + 0.5) * scale - 0.5
        anns['bbox'][0] *= x_scale  # (xmin, ymin, w, h)
        anns['bbox'][1] *= y_scale
        anns['bbox'][2] *= x_scale
        anns['bbox'][3] *= y_scale

        # modify geometry transformation parameters
        meta['offset'] *= scale  # (offset_x, offset_y)
        meta['scale'] *= scale   # scale
        # meta['valid_area'][:2] *= scale  # (xmin, ymin, w, h)
        # meta['valid_area'][2:] *= scale

        return image, anns, meta

class CenterPad(PreprocessAbstract):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image, anns, meta):
        # center padding
        w, h = image.size  # image is PIL image
        target_long = self.target_size
        left = int((target_long - w) / 2.0)
        top = int((target_long - h) / 2.0)
        ltrb = (
            left,
            top,
            target_long - w - left,
            target_long - h - top,
        )  # (left_offset, top_offset, right_offset, bottom_offset)

        # (124, 116, 104) is the mean of imagenet
        meanpixel = (124, 116, 104)
        if image.mode == 'RGBA':
            meanpixel = (124, 116, 104, 0)

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=meanpixel)

        # modify annotations
        anns['keypoints'][:, 0] += ltrb[0]  # N x 3, (x, y, is_visible) at each row
        anns['keypoints'][:, 1] += ltrb[1]
        anns['bbox'][0] += ltrb[0]  # (xmin, ymin, w, h)
        anns['bbox'][1] += ltrb[1]

        # modify geometry transformation parameters
        meta['offset'] -= ltrb[:2]  # (offset_x, offset_y), the pad image's coordinate frame relative to origin image coordinate frame
        # meta['valid_area'][:2] += ltrb[:2]  # (xmin, ymin, w, h)
        meta['pad_offset'] -= ltrb[:2]

        return image, anns, meta

class CoordinateNormalize(PreprocessAbstract):
    def __init__(self, normalize_keypoints=True, normalize_bbox=False):
        self.normalize_keypoints = normalize_keypoints
        self.normalize_bbox = normalize_bbox

    def __call__(self, image, anns, meta):  # note for this function we didn't record this transform parameters
        w, h = image.size  # image is PIL image

        if self.normalize_keypoints == True:
            # modify annotation
            # anns['keypoints'][:, 0] /= w
            # anns['keypoints'][:, 1] /= h

            anns['keypoints'][:, 0] = (anns['keypoints'][:, 0] / (w) - 0.5) * 2
            anns['keypoints'][:, 1] = (anns['keypoints'][:, 1] / (w) - 0.5) * 2


        if self.normalize_bbox == True:
            # anns['bbox'][0] /= w  # (xmin, ymin, w, h), ranges 0~1
            # anns['bbox'][1] /= h
            # anns['bbox'][2] /= w
            # anns['bbox'][3] /= h
            anns['bbox'][0] = (anns['bbox'][0] / (w) - 0.5) * 2  # (xmin, ymin, w, h), ranges -1~1
            anns['bbox'][1] = (anns['bbox'][1] / (h) - 0.5) * 2
            anns['bbox'][2] = anns['bbox'][2] / w * 2
            anns['bbox'][3] = anns['bbox'][3] / h * 2

        # meta['debug'].append(anns['keypoints'])
        # meta['debug'].append((w, h))

        return image, anns, meta


class HFlip(PreprocessAbstract):
    def __init__(self, swap=None):
        self.swap = swap  # swap is a function to swap the keypoint names when flipping

    def __call__(self, image, anns, meta):
        # plt.imshow(np.array(image))

        w, _ = image.size
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # modify annotations
        anns['keypoints'][:, 0] = w - 1 - anns['keypoints'][:, 0]
        if self.swap != None:  # swap keypoint names, since the left and right keypoints will exchange when flipping
            anns['keypoints'] = self.swap(anns['keypoints'])
        anns['bbox'][0] = w - 1 - (anns['bbox'][0] + anns['bbox'][2] - 1)   # (xmin, ymin, w, h)

        # modify geometry transformation parameters
        # meta['valid_area'][0] = w - 1 - (meta['valid_area'][0] + meta['valid_area'][2] - 1)  # (xmin, ymin, w, h)
        meta['hflip'] = True

        # plt.figure()
        # plt.imshow(np.array(image))
        # plt.show()

        return image, anns, meta


class RandomCrop1(PreprocessAbstract):
    def __init__(self, long_edge):  # random crop according to the given crop length
        self.long_edge = long_edge


    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, h = image.size
        padding = int(self.long_edge / 2.0)
        x_offset, y_offset = 0, 0
        if w > self.long_edge:
            x_offset = torch.randint(-padding, w - self.long_edge + padding, (1,))
            x_offset = torch.clamp(x_offset, min=0, max=w - self.long_edge).item()
        if h > self.long_edge:
            y_offset = torch.randint(-padding, h - self.long_edge + padding, (1,))
            y_offset = torch.clamp(y_offset, min=0, max=h - self.long_edge).item()

        # crop image
        new_w = min(self.long_edge, w - x_offset)
        new_h = min(self.long_edge, h - y_offset)
        ltrb = (x_offset, y_offset, x_offset + new_w, y_offset + new_h)
        image = image.crop(ltrb)

        anns['keypoints'][:, 0] -= x_offset
        anns['keypoints'][:, 1] -= y_offset
        anns['bbox'][0] -= x_offset
        anns['bbox'][1] -= y_offset

        return image, anns, meta

class RandomCrop(PreprocessAbstract):
    def __init__(self, crop_bbox=False):  # random aspect ratio and edge length for crop
        self.crop_bbox = crop_bbox  # if false, randomly crop image; else cropping the bbox directly

    def __call__(self, image, anns, meta):
        # plt.imshow(np.array(image))

        w, h = image.size

        # method 1
        # # random crop while maintaining all the keypoints in the valid region
        # inds_valid = anns['keypoints'][:, 2] > 0  # (x, y, is_visible)
        # keypoints_valid = anns['keypoints'][inds_valid]
        # # print('inds_valid', inds_valid)
        # xmin = np.min(keypoints_valid[:, 0])
        # ymin = np.min(keypoints_valid[:, 1])
        # xmax = np.max(keypoints_valid[:, 0])
        # ymax = np.max(keypoints_valid[:, 1])
        # xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # # print('ranges', xmin, ymin, xmax, ymax)

        # method 2
        # t1 = np.copy(anns['keypoints'])
        # t2 = np.copy(anns['bbox'])
        # judgex = np.logical_and(t1[:, 0] >= 0, t1[:, 0] < w)
        # judgey = np.logical_and(t1[:, 1] >= 0, t1[:, 1] < h)
        # if np.all(np.logical_and(judgex, judgey)) == False:
        #     print('wrong xy')

        xmin, ymin = int(anns['bbox'][0]), int(anns['bbox'][1])
        # xmax, ymax = int(anns['bbox'][0] + anns['bbox'][2]) - 1, int(anns['bbox'][1] + anns['bbox'][3]) - 1
        # for cropping, it is better to ensure the cropped region containing bbox, thus we perform inter ceiling
        xmax, ymax = int(anns['bbox'][0] + anns['bbox'][2]) + 1, int(anns['bbox'][1] + anns['bbox'][3]) + 1
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax >= w:
            xmax = w -1
        if ymax >= h:
            ymax = h -1

        offset_left, offset_top = 0, 0
        offset_right, offset_bottom = 0, 0
        if self.crop_bbox == False:  # random crop
            # compute the top-left and bottom-right coordinates (xmin, ymin, xmax, ymax) for random crop
            x1 = -xmin
            x2 = w - 1 - xmax
            y1 = -ymin
            y2 = h - 1 - ymax
            # if x1 > 0 or y1 > 0 or x2 < 0 or y2 < 0:
            #     print(w, h)
            #     print(anns['bbox'])
            #     print(xmin, xmax, ymin, ymax)
            #     print(x1, x2, y1, y2)

            # note that random.randint includes the upper-bound while torch.randint not
            # multiply 0.8 in order to avoid shifting too much
            offset_left = (int)(random.randint(int(x1), 0) * 0.8)
            offset_top = (int)(random.randint(int(y1), 0) * 0.8)
            offset_right = (int)(random.randint(0, int(x2)) * 0.8)
            offset_bottom = (int)(random.randint(0, int(y2)) * 0.8)

        # Here we let the  right+1 and bottom+1 since the coordinate system in PIL is a continuous system
        # where the (0, 0) is top-left and (w, h) is bottom-right. For example, if an image is 800 x 600 (w x h),
        # then its image rectangle is (0, 0, 800, 600)
        # (https: // pillow.readthedocs.io / en / stable / handbook / concepts.html  # coordinate-system)
        ltrb = (
            xmin + offset_left,
            ymin + offset_top,
            xmax + offset_right + 1,
            ymax + offset_bottom + 1,
        )  # (xmin, ymin, xmax, ymax)

        # crop
        image = image.crop(ltrb)

        # modify annotations
        anns['keypoints'][:, 0] -= ltrb[0]  # N x 3, (x, y, is_visible) at each row
        anns['keypoints'][:, 1] -= ltrb[1]
        anns['bbox'][0] -= ltrb[0]  # (x, y, w, h)
        anns['bbox'][1] -= ltrb[1]

        # modify geometry transformation parameters
        meta['offset'] += ltrb[:2]  # (offset_x, offset_y)
        # meta['valid_area'][:2] -= ltrb[:2]

        # plt.figure()
        # plt.imshow(np.array(image))
        # plt.show()

        # new_w, new_h = image.size
        # s1 = np.copy(anns['keypoints'])
        # s2 = np.copy(anns['bbox'])
        # judgexx = np.logical_and(s1[:, 0] >= 0, s1[:, 0] < new_w)
        # judgeyy = np.logical_and(s1[:, 1] >= 0, s1[:, 1] < new_h)
        # if np.all(np.logical_and(judgexx, judgeyy)) == False:
        #     print('wrong xy')
        # meta['debug'].append(t1)
        # meta['debug'].append((w,h))
        # meta['debug'].append(s1)
        # meta['debug'].append((new_w, new_h))

        return image, anns, meta

class RandomTranslation(PreprocessAbstract):
    def __init__(self, offset_xy=None):
        self.offset_xy = offset_xy

    def __call__(self, image, anns, meta):
        # plt.imshow(np.array(image))
        # print('translation called!')
        w, h = image.size
        # print('w, h', w, h)
        # image.save('1.jpg')

        # method 1
        # # random translation while maintaining all the keypoints in the valid region
        # inds_valid = anns['keypoints'][:, 2] > 0  # (x, y, is_visible)
        # keypoints_valid = anns['keypoints'][inds_valid]
        # # print('inds_valid', inds_valid)
        # xmin = np.min(keypoints_valid[:, 0])
        # ymin = np.min(keypoints_valid[:, 1])
        # xmax = np.max(keypoints_valid[:, 0])
        # ymax = np.max(keypoints_valid[:, 1])
        # xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # # print('ranges', xmin, ymin, xmax, ymax)

        # method 2
        xmin, ymin = int(anns['bbox'][0]), int(anns['bbox'][1])
        xmax, ymax = int(anns['bbox'][0] + anns['bbox'][2]) - 1, int(anns['bbox'][1] + anns['bbox'][3]) - 1
        # print('ranges', xmin, ymin, xmax, ymax)
        # if xmin < 0:
        #     xmin = 0
        # if ymin < 0:
        #     ymin = 0
        # if xmax >= w:
        #     xmax = w -1
        # if ymax >= h:
        #     ymax = h -1

        # compute the offset ranges for translation transform
        x1 = -xmin
        x2 = w - 1 - xmax
        y1 = -ymin
        y2 = h - 1 - ymax

        # method 2
        # print(int(x1), int(x2), int(y1), int(y2))
        random_offset_x = int(random.randint(int(x1), int(x2)) * 0.8)  # multiply 0.8 in order to avoid shifting too much
        random_offset_y = int(random.randint(int(y1), int(y2)) * 0.8)

        # print('traslation start')
        image_translate = self.translation(np.asarray(image), (random_offset_x, random_offset_y), (124, 116, 104))
        # print('traslation end')
        image = PIL.Image.fromarray(image_translate)
        # print('PIL end')

        # modify annotations
        anns['keypoints'][:, 0] += random_offset_x  # N x 3, (x, y, is_visible) at each row
        anns['keypoints'][:, 1] += random_offset_y
        anns['bbox'][0] += random_offset_x  # (x, y, w, h)
        anns['bbox'][1] += random_offset_y

        # modify geometry transformation parameters
        # meta['valid_area'][0] += random_offset_x  # (x, y, w, h)
        # meta['valid_area'][1] += random_offset_y

        # plt.figure()
        # plt.imshow(np.array(image))
        # plt.show()
        # print('translation end!')
        return image, anns, meta

    def translation(self, image, offset_xy, bordervalue):
        # print(image.shape, offset_xy, bordervalue)
        (h, w) = image.shape[:2]
        # ==========
        # don't know why the code of this line will cause below bug
        # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
        # M = np.float32([[1, 0, offset_xy[0]], [0, 1, offset_xy[1]]])
        # ==========
        M = cv2.getRotationMatrix2D((0,0), 0, 1.0)  # use this can circumvent the above error
        M[0, 0], M[0, 1], M[0, 2] = 1, 0, offset_xy[0]
        M[1, 0], M[1, 1], M[1, 2] = 0, 1, offset_xy[1]
        # ==========
        return cv2.warpAffine(image, M, (w, h), borderValue=bordervalue)


class RandomRotation(PreprocessAbstract):
    def __init__(self, max_rotate_degree=30):
        self.max_rotate_degree = max_rotate_degree

    def __call__(self, image, anns, meta):
        # plt.imshow(np.array(image))

        dice = random.random()
        degree = (dice - 0.5) * 2 * self.max_rotate_degree  # degree [-30,30]
        # (124, 116, 104) is the mean of imagenet
        meanpixel = (124, 116, 104)
        if image.mode == 'RGBA':
            meanpixel = (124, 116, 104, 0)
        img_rot, R = self.rotate_bound(np.asarray(image), degree, meanpixel)

        image = PIL.Image.fromarray(img_rot)

        # modify annotations
        num_keypoints = anns['keypoints'].shape[0]  # N x 3, (x, y, is_visible) at each row
        for k in range(num_keypoints):
            xy = anns['keypoints'][k, :2]
            new_xy = self.rotatepoint(xy, R)
            anns['keypoints'][k, :2] = new_xy
        anns['bbox'] = self.rotate_box(anns['bbox'], R)

        # modify geometry transformation parameters
        # meta['valid_area'] = self.rotate_box(meta['valid_area'], R)

        # plt.figure()
        # plt.imshow(np.array(image))
        # plt.show()

        return image, anns, meta

    @staticmethod
    def rotatepoint(p, R):
        point = np.zeros((3, 1))
        point[0] = p[0]
        point[1] = p[1]
        point[2] = 1

        new_point = R.dot(point)

        p[0] = new_point[0]

        p[1] = new_point[1]
        return p

    # The correct way to rotation an image
    # http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    # https: // docs.opencv.org / 3.4 / da / d6e / tutorial_py_geometric_transformations.html
    def rotate_bound(self, image, angle, bordervalue):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix, then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=bordervalue), M

    def rotate_box(self, bbox, R):
        """Input bounding box is of the form x, y, width, height."""
        four_corners = np.array([
            [bbox[0], bbox[1]],
            [bbox[0] + bbox[2] - 1, bbox[1]],
            [bbox[0], bbox[1] + bbox[3] - 1],
            [bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] -1],
        ])

        new_four_corners = []
        for i in range(4):
            xy = self.rotatepoint(four_corners[i], R)
            new_four_corners.append(xy)

        new_four_corners = np.array(new_four_corners)

        x = np.min(new_four_corners[:, 0])
        y = np.min(new_four_corners[:, 1])
        xmax = np.max(new_four_corners[:, 0])
        ymax = np.max(new_four_corners[:, 1])

        return np.array([x, y, xmax - x, ymax - y])


class RandomAffine(PreprocessAbstract):
    def __init__(self):
        pass

    def __call__(self, image, anns, meta):


        return image, anns, meta


class RandomPerspective(PreprocessAbstract):
    def __init__(self):
        pass

    def __call__(self, image, anns, meta):


        return image, anns, meta



class RandomGaussianNoise(PreprocessAbstract):
    def __init__(self, mean=0, std=30):
        # variation range is about 3*std
        self.mean = mean
        self.std = std

    def __call__(self, image, anns, meta):
        y = np.asarray(image, dtype=np.float)
        noise = np.random.normal(self.mean, self.std, size=y.shape)
        y = y + noise
        y = np.clip(y, a_min=0, a_max=255)
        y = y.astype(np.uint8)
        image = PIL.Image.fromarray(y)

        return image, anns, meta

class RandomSaltPepper(PreprocessAbstract):
    def __init__(self, ratio=0.1, grayscale=255):
        # Salt & Pepper noise is black-white noise
        # v > 255 * (1-ratio), white; otherwise, black
        self.ratio = ratio
        self.color = grayscale

    def __call__(self, image, anns, meta):
        y = np.asarray(image, dtype=np.float)
        h, w = y.shape[0:2]
        # method 1
        # noise = np.random.randint(0, 256, size=y.shape)  # randint is [low, high)
        # noise = np.where(noise>255*(1-self.ratio), 255, 0)
        # y = y + noise
        # y = np.clip(y, a_min=0, a_max=255)
        # method 2
        noise = np.random.rand(h, w)  # randint is [low, high)
        noise = np.where(noise > (1 - self.ratio), 1, 0)
        if len(y.shape) == 3:
            noise = noise.reshape(h, w, 1)
        y = y * (1-noise) + noise * self.color
        y = np.clip(y, a_min=0, a_max=255)
        y = y.astype(np.uint8)
        image = PIL.Image.fromarray(y)

        return image, anns, meta