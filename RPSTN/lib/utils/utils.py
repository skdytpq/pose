import sys
sys.path.append('')
import utils.Mytransforms_penn as Mytransforms
import utils.Mytransforms_JHMDB as Mytransforms_JHMDB
import torch.nn.functional as F
import math
import torch
import shutil
import time
import os
import random
from easydict import EasyDict as edict
import yaml
import numpy as np
import cv2
import time
from utils import penn_action_data as penn_action
from utils import Sub_JHMDB_data as sub_jhmdb


class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, iters, base_lr, gamma, step_size, weight_decay, policy='multi_step', multiple=[1]):

    if policy == 'fixed':
        lr = base_lr

    elif policy == 'step':
        lr = base_lr * (gamma ** (iters // step_size))

    elif policy == 'multi_step':
        lr = base_lr
        for s in range(len(step_size)):
            if (iters+1) == step_size[s]:
                lr *= weight_decay ** (s + 1)

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr

def save_checkpoint(state, is_best, filename='checkpoint', save_path='experiments/checkpoint/'):

    if is_best:
        torch.save(state, save_path + filename + '_best.pth.tar')

def Config(filename):

    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    return parser


def get_parameters(model, lr, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': lr},
            {'params': lr_2, 'lr': lr * 2.},
            {'params': lr_4, 'lr': lr * 4.},
            {'params': lr_8, 'lr': lr * 8.}]

    return params, [1., 2., 4., 8.]


def get_kpts(maps, img_h = 256.0, img_w = 256.0):

    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6[1:]:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts


def draw_paint(im, kpts, mapNumber):

           #       RED           GREEN           RED          YELLOW          YELLOW          PINK          GREEN
    colors = [[000,000,255], [000,255,000], [000,000,255], [255,255,000], [255,255,000], [255,000,255], [000,255,000],\
              [255,000,000], [255,255,000], [255,000,255], [000,255,000], [000,255,000], [000,000,255], [255,255,000], [255,000,000]]
           #       BLUE          YELLOW          PINK          GREEN          GREEN           RED          YELLOW           BLUE
    # draw points
    for k in kpts:
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=3, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        [Y1, X1] = kpts[limb[1]]

        if X0!=0 and Y0!=0 and X1!=0 and Y1!=0:
            if i<len(limbSeq)-4:
                cv2.line(cur_im, (Y0,X0), (Y1,X1), colors[i], 5)
            else:
                cv2.line(cur_im, (Y0,X0), (Y1,X1), [0,0,255], 5)

        im = cv2.addWeighted(im, 0.2, cur_im, 0.8, 0)

    cv2.imwrite('result/images/'+str(mapNumber)+'.png', im)


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def get_max_preds(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width      = batch_heatmaps.shape[3]

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx               = np.argmax(heatmaps_reshaped, 2)
    maxvals           = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx     = idx.reshape((batch_size, num_joints, 1))

    preds   = np.tile(idx, (1,1,2)).astype(np.float32)

    preds[:,:,0] = (preds[:,:,0]) % width
    preds[:,:,1] = np.floor((preds[:,:,1]) / width)

    pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
    pred_mask    = pred_mask.astype(np.float32)

    preds *= pred_mask

    return preds, maxvals


def getDataloader(dataset, train_dir, val_dir, test_dir, sigma, stride, workers, frame_memory, batch_size):
    # train_loader, val_loader, test_loader = None, None, None
    if dataset == 'Penn_Action':
        train_loader = torch.utils.data.DataLoader(
                                            penn_action.Penn_Action(train_dir, sigma, frame_memory, True,
                                            Mytransforms.Compose([
                                                              
                                                                Mytransforms.SinglePersonCrop(256),
                                                                Mytransforms.RandomRotate(40),
                                                                Mytransforms.TestResized(256),
                                                                Mytransforms.RandomHorizontalFlip(),
                                                                ])),
                                            batch_size  = batch_size, shuffle=True,
                                            num_workers = workers, pin_memory=True)
    
        val_loader   = torch.utils.data.DataLoader(
                                            penn_action.Penn_Action(val_dir, sigma, frame_memory, False,
                                            Mytransforms.Compose([
                                                                Mytransforms.SinglePersonCrop(256),
                                                                ])),
                                            batch_size  = batch_size, shuffle=False,
                                            num_workers = workers, pin_memory=True)

        test_loader = None

    elif dataset == 'Sub-JHMDB':
        trainAnnot, testAnnot = sub_jhmdb.get_train_test_annotation(train_dir)
        train_loader = torch.utils.data.DataLoader(
                                            sub_jhmdb.jhmdbDataset(trainAnnot, testAnnot, 5, 'train',
                                            Mytransforms_JHMDB.Compose([
                                                                Mytransforms_JHMDB.RandomResized(),
                                                                Mytransforms_JHMDB.RandomRotate(40),
                                                                Mytransforms_JHMDB.SinglePersonCrop(256),
                                                                Mytransforms_JHMDB.RandomHorizontalFlip(),
                                                                ])),
                                            batch_size  = batch_size, shuffle=True,
                                            num_workers = workers, pin_memory=True)
        val_loader = None
        test_loader = torch.utils.data.DataLoader(
                                            sub_jhmdb.jhmdbDataset(trainAnnot, testAnnot, 5, 'train',
                                            Mytransforms_JHMDB.Compose([
                                                                Mytransforms_JHMDB.SinglePersonCrop(256),
                                                                ])),
                                            batch_size  = batch_size, shuffle=False,
                                            num_workers = workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, dataset):
    from prettytable import PrettyTable
    mAPtable = PrettyTable(['Joints', 'mAP'])
    mPCKtable = PrettyTable(['Joints', 'PCK'])
    mPCKhtable = PrettyTable(['Joints', 'PCKh'])
    
    if dataset == "Penn_Action":

        print("------------ The mPCK of PennAction: %.2f%% --------------" % (mPCK*100))
        mPCKtable.add_row(['Head', PCK[0]])
        mPCKtable.add_row(['Shoulder', 0.5*(PCK[1]+PCK[2])])
        mPCKtable.add_row(['Elbow', 0.5*(PCK[3]+PCK[4])])
        mPCKtable.add_row(['Wrist', 0.5*(PCK[5]+PCK[6])])
        mPCKtable.add_row(['Hip', 0.5*(PCK[7]+PCK[8])])
        mPCKtable.add_row(['Knee', 0.5*(PCK[9]+PCK[10])])
        mPCKtable.add_row(['Ankle', 0.5*(PCK[11]+PCK[12])])
        mPCKtable.add_row(['Mean', mPCK])
        print(mPCKtable)

        # print("------------ The mPCKh of PennAction: %.2f%% -------------" % (mPCKh*100))
        # mPCKhtable.add_row(['Head', PCKh[0]])
        # mPCKhtable.add_row(['Shoulder', 0.5 * (PCKh[1] + PCKh[2])])
        # mPCKhtable.add_row(['Elbow', 0.5 * (PCKh[3] + PCKh[4])])
        # mPCKhtable.add_row(['Wrist', 0.5 * (PCKh[5] + PCKh[6])])
        # mPCKhtable.add_row(['Hip', 0.5 * (PCKh[7] + PCKh[8])])
        # mPCKhtable.add_row(['Knee', 0.5 * (PCKh[9] + PCKh[10])])
        # mPCKhtable.add_row(['Ankle', 0.5 * (PCKh[11] + PCKh[12])])
        # mPCKhtable.add_row(['Mean', mPCKh])
        # print(mPCKhtable)

    elif dataset == "Sub-JHMDB":
        print("------------ The mPCK of PennAction: %.2f%% --------------" % (mPCK*100))
        mPCKtable.add_row(['Head', PCK[2]])
        mPCKtable.add_row(['Shoulder', 0.5*(PCK[3]+PCK[4])])
        mPCKtable.add_row(['Elbow', 0.5*(PCK[7]+PCK[8])])
        mPCKtable.add_row(['Wrist', 0.5*(PCK[11]+PCK[12])])
        mPCKtable.add_row(['Hip', 0.5*(PCK[5]+PCK[6])])
        mPCKtable.add_row(['Knee', 0.5*(PCK[9]+PCK[10])])
        mPCKtable.add_row(['Ankle', 0.5*(PCK[13]+PCK[14])])
        mPCKtable.add_row(['Mean', mPCK])
        print(mPCKtable)

        # print("------------ The mPCKh of PennAction: %.2f%% -------------" % (mPCKh*100))
        # mPCKhtable.add_row(['Head', PCKh[2]])
        # mPCKhtable.add_row(['Shoulder', 0.5 * (PCKh[3] + PCKh[4])])
        # mPCKhtable.add_row(['Elbow', 0.5 * (PCKh[7] + PCKh[8])])
        # mPCKhtable.add_row(['Wrist', 0.5 * (PCKh[11] + PCKh[12])])
        # mPCKhtable.add_row(['Hip', 0.5 * (PCKh[5] + PCKh[6])])
        # mPCKhtable.add_row(['Knee', 0.5 * (PCKh[9] + PCKh[10])])
        # mPCKhtable.add_row(['Ankle', 0.5 * (PCKh[13] + PCKh[14])])
        # mPCKhtable.add_row(['Mean', mPCKh])
        # print(mPCKhtable)

def getOutImages(heat, input_var, img_path, outName):
    heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

    heat = heat.detach().cpu().numpy()

    heat = heat[0].transpose(1,2,0)


    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            for k in range(heat.shape[2]):
                if heat[i,j,k] < 0:
                    heat[i,j,k] = 0
                

    im = cv2.resize(cv2.imread(img_path[0]),(256, 256))

    heatmap = []
    for i in range(13):
        heatmap = cv2.applyColorMap(np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
        im_heat  = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
        cv2.imwrite('samples/WASPpose/heat/'+outName+'_'+str(i)+'.png', im_heat)
