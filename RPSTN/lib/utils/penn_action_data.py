# -*-coding:UTF-8-*-
import os
import pdb
import time
import scipy.io
import numpy as np
import random
import glob
import torch
import torch.utils.data as data
import scipy.misc
import cv2
import math 
import utils.Mytransforms_penn as Mytransforms
from torchvision import transforms
from PIL import Image


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def find_indices_srnn(frame_num1, frame_num2, seq_len, input_n=10):
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 4):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class Penn_Action(data.Dataset):
    def __init__(self, root_dir, sigma, frame_memory, is_train, transform=None):
        self.base = '../data/pose_data'
        self.width = 256
        self.height = 256
        self.transform = transform
        self.is_train = is_train
        self.sigma = sigma
        self.parts_num = 16 # joint 개수 
        self.seqTrain = frame_memory  # 5 로 지정
        self.min_scale = 0.8
        self.max_scale = 1.4
        self.max_degree = 40
        self.heatmap_size = 64
        self.stride = 4

        self.root_dir = 'pose_data/'#root_dir
        self.label_dir = root_dir + 'labels'
        self.frame_dir = os.path.join(self.base,'frames')#root_dir + 'frames' 
        self.train_dir = os.path.join(self.base,'train')
        self.val_dir = os.path.join(self.base,'test') 

        if self.is_train is True:
            self.data_dir = '../data/pose_data/train' # root_dir + 'train/'
            self.p_scale = 1
            self.p_rotate = 1
            self.p_flip = 1
        else:
            self.data_dir ='../data/pose_data/test' #root_dir + 'test/'
            self.p_scale = 0
            self.p_rotate = 0
            self.p_flip = 0

        self.frames_data = os.listdir(self.data_dir)

    def __getitem__(self, index):
        frames = self.frames_data[index]
        if 'DS_Store' in  os.path.join(self.data_dir, frames):
            pass
        data = np.load(os.path.join(self.data_dir, frames), allow_pickle=True)[0]
        nframes = data['nframes'][0]    # 151 , T : 고정적이지 않음
        framespath = data['framepath']
        dim = data['dimensions'][0] # [360,480] , [H,W,T]
        x = data['x']          # 151 * 13
        y = data['y']          # 151 * 13
        visibility = data['visibility'] # 151 * 13
        anno_bboxes = data['bbox']
        img_paths = []
        
        if self.is_train:
            start_index = np.random.randint(1, nframes - 1 - self.seqTrain + 1)
        else:
            SEED = 10480 + index
            rng = np.random.RandomState(SEED)
            start_index = rng.randint(1, nframes - 1 - self.seqTrain + 1)
        label_size = self.heatmap_size

        l = np.zeros((self.seqTrain, label_size, label_size, self.parts_num))
        label_map = torch.zeros(self.seqTrain, self.parts_num, label_size, label_size)

        images = torch.zeros(self.seqTrain, 3, self.height, self.width)  # [3,256,256]
        boxes = torch.zeros(self.seqTrain, 5, label_size, label_size)  # [5,64,64]
        label = np.zeros((self.seqTrain, self.parts_num, 3)) # 프레임 길이, 운동 종목 수 , 3(x,y,visibility)
     
        kps = np.zeros((self.seqTrain, self.parts_num + 5, 3)) 
        bbox = np.zeros((self.seqTrain, 4))
        person_box = np.zeros((self.seqTrain, 4, 3))
        kp = np.zeros((self.seqTrain,self.parts_num,2))
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
            # 수정작업
            #img_path = os.path.join(framespath.replace('_', ''), '%06d' % (start_index + i + 1) + '.jpg')
            img_path = os.path.join(f'{self.frame_dir}/{framespath.split(".")[0]}', '%06d' % (start_index + i + 1) + '.jpg')
            img_paths.append(img_path) 
            img = cv2.imread(img_path).astype(dtype=np.float32)  # Image
            # read label [5,16,3]
            label[i, :, 0] = x[start_index + i] # ground truth 지정을 위해 각각 x, y visibilityf를 매핑
            label[i, :, 1] = y[start_index + i]
            label[i, :, 2] = visibility[start_index + i]  # 1 * 13
            bbox[i, :]     = data['bbox'][start_index + i]  #
            # make the joints not in the figure vis=-1(Do not produce label) 
            for part in range(0, self.parts_num):  # for each part
                if self.isNotOnPlane(label[i, part, 0], label[i, part, 1], dim[1], dim[0]):
                    label[i, part, 2] = -1

            
            kps[i, :16, :] = label[i] # kps 에서 :16 까지는 각 Joint 의 x,y visibility 를 다룬다
            # label[i]는 각 시점에서 x , y visibility 를 갖고있다.
            center_x = int(self.width/2)
            center_y = int(self.height/2)
            center   = [center_x, center_y]

            kps[i, 16] = [int((bbox[i,0]+bbox[i,2])/2),int((bbox[i,1]+bbox[i,3])/2),1]
            kps[i, 17] = [bbox[i,0],bbox[i,1],1] 
            kps[i, 18] = [bbox[i,0],bbox[i,3],1] 
            kps[i, 19] = [bbox[i,2],bbox[i,1],1] 
            kps[i, 20] = [bbox[i,2],bbox[i,3],1] 
            # kps 는 일괄적으로 이미지를 crop 하기 위해 각 x,y, visibility, bbox 좌표를 모은 것 
            img, kps[i], center = self.transform(img, kps[i], center, randoms)
            box  = kps[i, -5:]
            kpts = kps[i, :self.parts_num]
            label[i, :, :] = kpts
            person_box[i] = box[1:]

            img = cv2.resize(img, (256, 256))
            images[i, :, :, :] = transforms.ToTensor()(img)
            heatmap = np.zeros((label_size, label_size, self.parts_num), dtype=np.float32)
            # 64 X 64 X 13
            tmp_size = self.sigma * 3
            for k in range(self.parts_num):
                xk = int(kpts[k][0] / self.stride)
                yk = int(kpts[k][1] / self.stride)

                ul = [int(xk - tmp_size), int(yk - tmp_size)]
                br = [int(xk + tmp_size + 1), int(yk + tmp_size + 1)]

                if ul[0] >= self.heatmap_size or ul[1] >= self.heatmap_size \
                    or br[0] < 0 or br[1] < 0:
                    label[i, k, -1] = 0
                    continue # label size -> heatmap size
                heat_map = guassian_kernel(size_h=label_size, size_w=label_size, center_x=xk, center_y=yk, sigma=self.sigma)
                heat_map[heat_map > 1] = 1
                heat_map[heat_map < 0.0099] = 0
                heatmap[:, :, k] = heat_map
                 
            l[i] = heatmap
            label_map[i] = transforms.ToTensor()(heatmap)
            # 가우시안 커널을 통해 label_map 즉, label의 관절 히트맵을 구한 것
            kp[i,:,:] = kps[i,:self.parts_num,:2]
        return images, label_map, label, img_paths, person_box, start_index,kp # keypoints 좌표 
# 이미지, 히트맵, 라벨(실제 참 값인 관절 좌표값) , 이미지 경로, 사람 바운딩 박스, 시작 지점

    def isNotOnPlane(self, x, y, width, height):
        notOn = x < 0.001 or y < 0.001 or x > width or y > height
        return notOn


    def __len__(self):
        return len(self.frames_data)
