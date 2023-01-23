import numpy as np
import torch
import cv2
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from h5py import File
import utils.Mytransforms_JHMDB as Mytransforms

import os
import scipy.io
import statistics
import random
from six.moves import xrange
import fnmatch
import json
import glob

# borrowed from: https://github.com/lawy623/LSTM_Pose_Machines/blob/master/dataset/JHMDB/JHMDB_PreData.m
# order
# 0: neck    1:belly   2: face
# 3: right shoulder  4: left shoulder
# 5: right hip       6: left hip
# 7: right elbow     8: left elbow
# 9: right knee      10: left knee
# 11: right wrist    12: left wrist
# 13: right ankle    14: left ankle

def get_train_test_annotation(dataRoot):

    subFolder = os.path.join(dataRoot, 'sub-splits')
    imageFolder = os.path.join(dataRoot, 'Rename_Images')
    maskFolder = os.path.join(dataRoot, 'puppet_mask')
    poseFolder = os.path.join(dataRoot, 'joint_positions')

    # baselineFolder = os.path.join(dataRoot, 'your baseline folder')

    totTXTlist = os.listdir(subFolder)

    trainAnnot = []
    testAnnot = []
    for i in range(0, len(totTXTlist)):
        filename = os.path.join(subFolder, totTXTlist[i])
        action = totTXTlist[i].split('_test_')[0]

        with open(filename) as f:
            content = f.readlines()

        for t in range(0, len(content)):

            folder_to_use = content[t].split('\n')[0].split('.avi')[0]
            traintest = content[t].split('\n')[0].split('.avi')[1]   # 1: train; 2: test

            imgPath = os.path.join(imageFolder, action, folder_to_use)
            posePath = os.path.join(poseFolder, action, folder_to_use)
            maskPath = os.path.join(maskFolder, action, folder_to_use)


            annot = scipy.io.loadmat(os.path.join(posePath, 'joint_positions'))
            bbox = scipy.io.loadmat(os.path.join(maskPath, 'Bbox.mat'))['Bbox']
            mask = scipy.io.loadmat(os.path.join(maskPath, 'puppet_mask.mat'))['part_mask']
            # print(imgPath)
            if int(traintest) == 1:
                dicts = {'imgPath': imgPath, 'annot': annot, 'Bbox': bbox, 'mask': mask}
                # dicts = {'imgPath': imgPath, 'annot': annot, 'Bbox': bbox, 'mask': mask, 'baseline': None}

                trainAnnot.append(dicts)
                
            else:
                dicts = {'imgPath': imgPath, 'annot': annot, 'Bbox': bbox, 'mask': mask}
                testAnnot.append(dicts)
        # print(len(trainAnnot), len(testAnnot))

    return trainAnnot, testAnnot


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

class jhmdbDataset(data.Dataset):
    def __init__(self, trainAnnot, testAnnot, T, split, transform=None):
        # if_occ: occlusion ratio for robust comparison experiment
        
        self.trainSet = trainAnnot#[0:600]
        self.testSet = testAnnot
        self.valSet = None #trainAnnot[600:]
        self.seqTrain = T
        self.split = split

        self.sigma = 2
        self.numJoint = 15
        self.input_h = 256
        self.input_w = 256
        self.heatmap_size = 64
        self.stride = 4
        self.min_scale = 0.8
        self.max_scale = 1.4
        self.max_degree = 40
        # self.if_occ = if_occ
        self.transform = transform

        if self.split == 'train':
            self.dataLen = len(self.trainSet)
            self.p_scale = 1
            self.p_rotate = 1
            self.p_flip = 1
        elif self.split == 'val':
            self.dataLen = len(self.valSet)
            self.p_scale = 0
            self.p_rotate = 0
            self.p_flip = 0
        else:
            self.dataLen = len(self.testSet)
            self.p_scale = 0
            self.p_rotate = 0
            self.p_flip = 0

        numData = len(self.trainSet)
        allSkeleton = []
        for i in range(0, numData):
            skeleton = self.trainSet[i]['annot']['pos_img']
            allSkeleton.append(skeleton)

        allSkeleton = np.concatenate((allSkeleton), 2)

    def __len__(self):
        return self.dataLen
        # return 10               # TO DEBUG

    def read_annot(self, annotSet):
        imgPath = annotSet['imgPath']
        Bbox = annotSet['Bbox']
        skeleton = annotSet['annot']['pos_img'].transpose(2, 1, 0)     # 2 x 15 x T ---> T x 15 x 2
        img_mask = annotSet['mask']

        return imgPath, Bbox, skeleton

    def data_to_use(self, imgPath, Bbox, gtSkeleton):
        nframes = Bbox.shape[0]
        # print('number of frames:', Bbox.shape)
        random.seed(1234567890)
        useLen = self.seqTrain


        # if nframes > useLen:
        start_index = random.randint(0, nframes - useLen)


        label_size = self.heatmap_size

        l = np.zeros((self.seqTrain, label_size, label_size, self.numJoint))
        label_map = torch.zeros(self.seqTrain, self.numJoint, label_size, label_size)

        images = torch.zeros(self.seqTrain, 3, self.input_h, self.input_w)  # [3,256,256]
        label = np.zeros((self.seqTrain, self.numJoint, 3))
    
        kps = np.zeros((self.seqTrain, self.numJoint + 5, 3))
        bbox = np.zeros((self.seqTrain, 4))
        person_box = np.zeros((self.seqTrain, 4, 3))

        img_paths = []
        scale_factor = 0
        rotate_angle = 0
        flip_factor = 0

        if random.random() < self.p_scale:
            scale_factor = random.uniform(self.min_scale, self.max_scale)
        if random.random() < self.p_rotate:
            rotate_angle = -self.max_degree + 2 * self.max_degree * random.random()
        if random.random() < self.p_flip:
            flip_factor = random.random()
        randoms = [scale_factor, rotate_angle, flip_factor]

        # build data set--------
        for i in range(self.seqTrain):
            img_path = os.path.join(imgPath, '%05d' % (start_index + i + 1) + '.png')
            img_paths.append(img_path)
            # print(img_path)
            img = cv2.imread(img_path).astype(dtype=np.float32)  # Image
            # read label
            label[i, :, 0] = gtSkeleton[start_index + i, :, 0]
            label[i, :, 1] = gtSkeleton[start_index + i, :, 1]
            label[i, :, 2] = np.ones((1, self.numJoint)) #visibility[start_index + i]  # 1 * 13
            # bbox[i, :]     = Bbox[start_index + i] 
            bbox[i, 0] = Bbox[start_index + i, 0] + 6
            bbox[i, 1] = Bbox[start_index + i, 1] + 2 
            bbox[i, 2] = Bbox[start_index + i, 2] - 2
            bbox[i, 3] = Bbox[start_index + i, 3]
            # make the joints not in the figure vis=-1(Do not produce label)
            for part in range(0, self.numJoint):  # for each part
                if self.isNotOnPlane(label[i, part, 0], label[i, part, 1], self.input_h, self.input_w):
                    label[i, part, 2] = -1
            
            kps[i, :15, :] = label[i]
            
            center_x = int(self.input_w/2)
            center_y = int(self.input_h/2)
            center   = [center_x, center_y]

            kps[i, 15] = [int((bbox[i,0]+bbox[i,2])/2),int((bbox[i,1]+bbox[i,3])/2),1]
            kps[i, 16] = [bbox[i,0],bbox[i,1],1] 
            kps[i, 17] = [bbox[i,0],bbox[i,3],1] 
            kps[i, 18] = [bbox[i,2],bbox[i,1],1] 
            kps[i, 19] = [bbox[i,2],bbox[i,3],1] 
            # print(img.shape)
            # img, kps[i], center = self.transform(img, kps[i], center, randoms)
            img, kps[i] = self.transform(img, kps[i], center, randoms)
            box  = kps[i, -5:]
            kpts = kps[i, :self.numJoint]
            label[i, :, :] = kpts
            person_box[i] = box[1:]

            # img = cv2.resize(img, (256, 256))
            images[i, :, :, :] = transforms.ToTensor()(img)
            heatmap = np.zeros((label_size, label_size, self.numJoint), dtype=np.float32)
            tmp_size = self.sigma * 3
            for k in range(self.numJoint):
                # resize from 368 to 46
                xk = int(kpts[k][0] / self.stride)
                yk = int(kpts[k][1] / self.stride)

                ul = [int(xk - tmp_size), int(yk - tmp_size)]
                br = [int(xk + tmp_size + 1), int(yk + tmp_size + 1)]

                if ul[0] >= self.heatmap_size or ul[1] >= self.heatmap_size \
                    or br[0] < 0 or br[1] < 0:
                    label[i, k, -1] = 0
                    continue
                heat_map = guassian_kernel(size_h=label_size, size_w=label_size, center_x=xk, center_y=yk, sigma=self.sigma)
                heat_map[heat_map > 1] = 1
                heat_map[heat_map < 0.0099] = 0
                heatmap[:, :, k] = heat_map
                
            l[i] = heatmap
            label_map[i] = transforms.ToTensor()(heatmap)

        return images, label_map, label, img_paths, person_box, start_index


    def isNotOnPlane(self, x, y, width, height):
        notOn = x < 0.001 or y < 0.001 or x > width or y > height
        return notOn


    def __len__(self):
        return self.dataLen


    def __getitem__(self, idx):
        if self.split == 'train':
            annotSet = self.trainSet[idx]
        elif self.split == 'val':
            annotSet = self.valSet[idx]
        else:
            annotSet = self.testSet[idx]

        imgPath, Bbox, gtSkeleton = self.read_annot(annotSet)
        # sequence_to_use, Bbox_to_use, imgSequence_to_use, mask_idx, nframes, idx = self.data_to_use(imgPath, Bbox, gtSkeleton)
        images, label_map, label, img_paths, person_box, start_index = self.data_to_use(imgPath, Bbox, gtSkeleton)
        # print(images.shape, label.shape, label_map.shape, person_box.shape)
        dicts = {'imgSequence_to_use': images, 'Bbox_to_use': person_box,
                 'heatmap_to_use': label_map, 'coords_to_use': label, 'img_path': img_paths,
                 'randomInd':start_index}

        return images, label_map, label, img_paths, person_box, start_index