# -*-coding:UTF-8-*-
from __future__ import division
import torch
import random
import numpy as np
import numbers
import collections
import cv2
import time


def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def to_tensor(pic):
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    return img.float()


def resize(img, kpt, center, ratio):
    # print(img.shape)
    if not (isinstance(ratio, numbers.Number) or (isinstance(ratio, collections.Iterable) and len(ratio) == 2)):
        raise TypeError('Got inappropriate ratio arg: {}'.format(ratio))
    
    h, w, _ = img.shape
    if w < 64:
        img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - w, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        w = 64
    
    if isinstance(ratio, numbers.Number):
        num = len(kpt)
        for i in range(num):
            kpt[i][0] *= ratio
            kpt[i][1] *= ratio
        center[0] *= ratio
        center[1] *= ratio
        return cv2.resize(img, (0, 0), fx=ratio, fy=ratio), kpt, center
    else:
        num = len(kpt)
        for i in range(num):
            kpt[i][0] *= ratio[0]
            kpt[i][1] *= ratio[1]
        center[0] *= ratio[0]
        center[1] *= ratio[1]
    return np.ascontiguousarray(cv2.resize(img,(int(img.shape[0]*ratio[0]),int(img.shape[1]*ratio[1])),interpolation=cv2.INTER_CUBIC)), kpt, center


class RandomResized(object):
    def __init__(self, scale_min=0.8, scale_max=1.4):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, img, kpt, center, randoms):
        ratio = randoms[0] #random.uniform(self.scale_min, self.scale_max)

        return resize(img, kpt, center, ratio)


class TestResized(object):
    def __init__(self, size):
        assert (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        # print(img.shape)
        height, width, _ = img.shape
        
        return (output_size[0] * 1.0 / height, output_size[1] * 1.0 / width)

    def __call__(self, img, kpt, center, randoms):
        # print(img.shape)
        ratio = self.get_params(img, self.size)
        img, kpt, center = resize(img, kpt, center, ratio)
        # print('In TestResized: ', img.shape)
        return img, kpt, center


def rotate(img, kpt, center, degree):
    height, width, _ = img.shape

    img_center = (width / 2.0 , height / 2.0)
    rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)
    cos_val = np.abs(rotateMat[0, 0])
    sin_val = np.abs(rotateMat[0, 1])
    new_width = int(height * sin_val + width * cos_val)
    new_height = int(height * cos_val + width * sin_val)
    rotateMat[0, 2] += (new_width / 2.) - img_center[0]
    rotateMat[1, 2] += (new_height / 2.) - img_center[1]

    img = cv2.warpAffine(img, rotateMat, (new_width, new_height), borderValue=(128, 128, 128))

    num = len(kpt)
    for i in range(num):
        if kpt[i][2]==0:
            continue
        x = kpt[i][0]
        y = kpt[i][1]
        p = np.array([x, y, 1])
        p = rotateMat.dot(p)
        kpt[i][0] = p[0]
        kpt[i][1] = p[1]

    x = center[0]
    y = center[1]
    p = np.array([x, y, 1])
    p = rotateMat.dot(p)
    center[0] = p[0]
    center[1] = p[1]
    # print('In rotate: ', img.shape)
    return np.ascontiguousarray(img), kpt, center


class RandomRotate(object):
    def __init__(self, max_degree):
        assert isinstance(max_degree, numbers.Number)
        self.max_degree = max_degree

    @staticmethod
    def get_params(max_degree):
        degree = random.uniform(-max_degree, max_degree)
        return degree

    def __call__(self, img, kpt, center, randoms):
   
        degree = randoms[1] #self.get_params(self.max_degree)
        img, kpt, center = rotate(img, kpt, center, degree)
        # print('In RandomRotate: ', img.shape)
        return img, kpt, center

def scale(image, kpt, center, f_xy):
    (h, w, _) = image.shape
    h, w = int(h * f_xy[1]), int(w * f_xy[0])
    # print(h, w, f_xy)
    # print('In scale: ', image.shape, f_xy)
    image = cv2.resize(image, (0, 0), fx=f_xy[0], fy=f_xy[1]).astype(np.uint8)

    num = len(kpt)
    for i in range(num):
        kpt[i][0] *= f_xy[0]
        kpt[i][1] *= f_xy[1]
    center[0] *= f_xy[0]
    center[1] *= f_xy[1]

    kpt[:15, 0] = np.clip(kpt[:15, 0], 0, w)
    kpt[:15, 1] = np.clip(kpt[:15, 1], 0, h)
    return image, kpt, center

def crop(image, kpt, center, length):
    x, y = kpt[:15, 0], kpt[:15, 1]
    bbox = np.array([kpt[16, 0], kpt[16, 1], kpt[19, 0], kpt[19, 1]])
    x, y, bbox = x.astype(np.int), y.astype(np.int), bbox.astype(np.int)

    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    # print(w, h)
    if w == 0 or h == 0:
        # print('The width or height is 0: ', image.shape)
        ratio_h, ratio_w = float(length) / image.shape[0], float(length) / image.shape[1] 
        cropped, kpt, center = resize(image, kpt, center, [ratio_h, ratio_w])
    else:
        # Crop image to bbox
        image = image[y_min:y_min + h, x_min:x_min + w, :]

        # Crop joints and bbox
        x -= x_min
        y -= y_min
        bbox = np.array([0, 0, x_max - x_min, y_max - y_min])

        # Scale to desired size 
        side_length = [w, h]
        
        f_xy = [float(length) / float(side_length[0]), float(length) / float(side_length[1])]

        # image, bbox, x, y = scale(image, bbox, x, y, f_xy)
        kps = kpt.copy()
        kps[:15, 0], kps[:15, 1] = x, y

        # print('In crop: ', image.shape)
        image, kps, center = scale(image, kps, center, f_xy)
        x, y = kps[:15, 0], kps[:15, 1]
        bbox = np.array([kps[16, 0], kps[16, 1], kps[19, 0], kps[19, 1]])

        new_w, new_h = image.shape[1], image.shape[0]
        cropped = np.zeros((length, length, image.shape[2]))

        dx = length - new_w
        dy = length - new_h
        x_min, y_min = int(dx / 2.), int(dy / 2.)
        x_max, y_max = x_min + new_w, y_min + new_h

        cropped[y_min:y_max, x_min:x_max, :] = image
        x += x_min
        y += y_min

        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)

        bbox += np.array([x_min, y_min, 0, 0])
        bbox  = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
        kpt[:15, 0], kpt[:15, 1] = x, y
        kpt[16, 0], kpt[16, 1], kpt[19, 0], kpt[19, 1] = bbox
        kpt[15, 0], kpt[15, 1] = (bbox[0]+bbox[2]) / 2, (bbox[1]+bbox[3]) / 2
        center = [length/2, length/2]

    return np.ascontiguousarray(cropped), kpt, center


class SinglePersonCrop(object):
    def __init__(self, size, center_perturb_max=5):
        assert isinstance(size, numbers.Number)
        # self.length = 256
        self.size = size  # (w, h) (368, 368)
        self.center_perturb_max = center_perturb_max

    @staticmethod
    def get_params(img, center, output_size, center_perturb_max):
        return int(round(center[0] - output_size[0] / 2)), int(round(center[1] - output_size[1] / 2))


    def __call__(self, img, kpt, center, random):
        # offset_left, offset_up = self.get_params(img, center, self.size, self.center_perturb_max)
        img, kpt, center = crop(img, kpt, center, self.size)
        # print('In SinglePersonCrop: ', img.shape)
        return img, kpt, center


def hflip(img, kpt, center):

    height, width, _ = img.shape
    swap_pair = [[13, 24], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]

    # img = img[:, ::-1, :]
    img = cv2.flip(img, 1)
    num = len(kpt)
    for i in range(num):
        if kpt[i][2] == 1:
            kpt[i][0] = width - 1 - kpt[i][0]
    center[0] = width - 1 - center[0]

    for x in swap_pair:
        temp_point = kpt[x[0]].copy()
        kpt[x[0]] = kpt[x[1]].copy()
        kpt[x[1]] = temp_point
    return np.ascontiguousarray(img), kpt, center


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, img, kpt, center, randoms):
        if randoms[2] > self.prob:
            return hflip(img, kpt, center)
        # print('In RandomHorizontalFlip: ', img.shape)
        return img, kpt, center


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpt, center, randoms):
        # time.sleep(0.5)
        for t in self.transforms:
            if isinstance(t, RandomResized):
                img, kpt, center = t(img, kpt, center, randoms)
            else:
                img, kpt, center = t(img, kpt, center, randoms)

        return img, kpt, center
