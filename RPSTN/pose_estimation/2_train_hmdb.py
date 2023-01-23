# -*-coding:UTF-8-*-
import argparse
import _init_paths
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import cv2
import os
import models
import math
import pdb
import shutil

# sys.path.append("..")
from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import save_checkpoint as save_checkpoint
from utils.utils import printAccuracies as printAccuracies
from utils.utils import guassian_kernel as guassian_kernel
from utils.utils import get_parameters  as get_parameters
from utils import Mytransforms_update_JHMDB as  Mytransforms 
from utils.utils import getDataloader as getDataloader
from utils.utils import AverageMeter as AverageMeter
from utils import evaluate as evaluate
from utils.vis import *

from core.config import config
from core.loss import MSESequenceLoss, JointsMSELoss
from tqdm import tqdm

import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary
from tensorboardX import SummaryWriter

from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

class Trainer(object):
    def __init__(self, args, is_train, is_visual):
        self.args = args
        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.model_arch = args.model_arch
        self.dataset = args.dataset
        self.frame_memory = args.frame_memory   # 每个视频帧数 T
        self.writer = args.writer
        self.gpus = [int(i) for i in config.GPUS.split(',')]
        self.is_train = is_train
        self.is_visual = is_visual

        self.workers = 8
        self.weight_decay = 0.1
        self.momentum = 0.9
        self.batch_size = 8
        self.lr = 0.0005
        self.gamma = 0.333
        self.step_size = [4, 8, 15, 25, 40, 80]#13275
        self.sigma = 2
        self.stride = 4
        self.heatmap_size = 64

        cudnn.benchmark = True

        if self.dataset ==  "Sub-JHMDB":
            self.numClasses = 15
            self.test_dir = None

        self.train_loader, self.val_loader, self.test_loader = getDataloader(self.dataset, self.train_dir, \
                                                                self.val_dir, self.test_dir, self.sigma, self.stride, \
                                                                self.workers, self.frame_memory, \
                                                                self.batch_size)

        model = models.hmdb_dkd_net.get_dkd_net(config, self.is_visual, is_train=True if self.is_train else False)
        self.model = torch.nn.DataParallel(model, device_ids=self.gpus).cuda()
        self.criterion = MSESequenceLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.iters = 0

        if self.args.pretrained is not None:
            checkpoint = torch.load(self.args.pretrained)
            p = checkpoint['state_dict']
            if self.dataset == "Sub-JHMDB":
                prefix = 'invalid'
            state_dict = self.model.state_dict()
            model_dict = {}

            for k,v in p.items():
                if k in state_dict:
                    if not k.startswith(prefix):                                
                        model_dict[k] = v

            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict)
            print('Loading Successfully')
            
        self.isBest = 0
        self.bestPCK  = 0
        self.bestPCKh = 0
        self.best_epoch = 0


    def training(self, epoch):
        print('Start Training....')
        train_loss = 0.0
        self.model.train()
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)
        for i, (input, heatmap, label, img_path, bbox, start_index) in enumerate(tbar):
            learning_rate = adjust_learning_rate(self.optimizer, epoch, self.lr, weight_decay=self.weight_decay, policy='multi_step',
                                                 gamma=self.gamma, step_size=self.step_size)

            vis = label[:, :, :, -1]
            vis = vis.view(-1, self.numClasses, 1)

            input_var = input.cuda()
            heatmap_var = heatmap.cuda()

            heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).cuda()

            losses = {}
            loss = 0
            start_model = time.time()
            heat = self.model(input_var, heatmap_var)
            losses = self.criterion(heat, heatmap_var)

            loss += losses #+ 0.5 * relation_loss)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_acc = evaluate.cal_train_acc(heat, heatmap_var)      

            tbar.set_postfix(loss='%.4f'%(train_loss / self.batch_size), acc='%.2f'%(train_acc * 100))


    def validation(self, epoch):
        print('Start Testing....')
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0

        pck_thred = 0.2
        AP = np.zeros(self.numClasses)
        PCK = np.zeros(self.numClasses)
        PCKh = np.zeros(self.numClasses)
        count = np.zeros(self.numClasses)

        idx = []
        cnt = 0
        preds = []
        for i, (input, heatmap, label, img_path, bbox, start_index) in enumerate(tbar):
            # np.save('result/images/img_path.npy', img_path)
            cnt += 1
            idx.append(start_index)
            input_var = input.cuda()
            heatmap_var = heatmap.cuda()

            self.optimizer.zero_grad()

            heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).cuda()
            vis = label[:, :, :, -1]
            vis = vis.view(-1, self.numClasses, 1)

            losses = {}
            loss   = 0

            start_model = time.time()
            heat = self.model(input_var, heatmap_var)
            losses = self.criterion(heat, heatmap_var)
            loss  += losses.item() #+ 0.5 * relation_loss.item()
            
            b, t, c, h, w = input.shape
           
            if self.is_visual:
                file_name = 'result/heats/{}_batch.jpg'.format(i)

                input = input.view(-1, c, h, w)
                heat = heat.view(-1, self.numClasses, heat.shape[-2], heat.shape[-1])
                
            input, heat = input.view(b, t, c, h, w).contiguous(), heat.view(b, t, self.numClasses, heat.shape[-2], heat.shape[-1]).contiguous()

            for j in range(heat.size(0)): #self.frame_memory):
                acc, acc_PCK, acc_PCKh, cnt, pred, visible = evaluate.accuracy(heat[j].detach().cpu().numpy(),\
                                            heatmap_var[j].detach().cpu().numpy(), pck_thred, 0.5, self.dataset, bbox[j], normTorso=False)
                preds.append(pred)
                for k in range(self.numClasses):
                    if visible[k] == 1:
                        if k != 0 or k != 1:
                            AP[k] = (AP[k] * count[k] + acc[k]) / (count[k]+1)
                            PCK[k] = (PCK[k] * count[k] + acc_PCK[k]) / (count[k]+1)
                            PCKh[k] = (PCKh[k]* count[k] + acc_PCKh[k]) / (count[k]+1)
                            count[k] += 1
                        
            mAP = AP[:].sum()/(self.numClasses)
            mPCK = PCK[:].sum()/(self.numClasses)
            mPCKh = PCKh[:].sum()/(self.numClasses)


            val_loss += loss

            tbar.set_postfix(valoss='%.6f' % (val_loss / self.batch_size), mPCK=mPCK)

        printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, self.dataset)
        
        

        PCKhAvg = PCKh.sum()/(self.numClasses-2)
        PCKAvg  =  PCK.sum()/(self.numClasses-2)
        times = time.strftime('%Y%m%d', time.localtime())

        if mPCK > self.isBest:
            self.isBest = mPCK
            source_model_save_path = self.args.source_model_save_path + self.args.model_name
            if not os.path.exists(source_model_save_path):
                os.makedirs(source_model_save_path)
            if self.is_train is True:
                shutil.copy2('lib/models/dkd_net.py', source_model_save_path)
                np.save('experiments/best_index', start_index)
                save_checkpoint({'state_dict': self.model.state_dict()}, self.isBest, self.args.model_name+'_'+times, self.args.model_save_path)

        if mPCKh > self.bestPCKh:
            self.bestPCKh = mPCKh
        if mPCK > self.bestPCK:
            self.bestPCK = mPCK
            self.best_epoch = epoch

        print("Best epoch: %d; PCK = %2.2f%%; PCKh = %2.2f%%" % (self.best_epoch, self.bestPCK*100,self.bestPCKh*100))


    def test(self, epoch):
        print('Start Testing....')
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        val_loss = 0.0
        pck_thred = 1

        AP = np.zeros(self.numClasses)
        PCK = np.zeros(self.numClasses)
        PCKh = np.zeros(self.numClasses)
        count = np.zeros(self.numClasses)
        res_pck = np.zeros(self.numClasses+1)

        idx = []
        cnt = 0
        preds = []
        for i, (input, heatmap, label, img_path, bbox, start_index) in enumerate(tbar):
            # np.save('result/images/img_path.npy', img_path)
            cnt += 1
            idx.append(start_index)
            input_var = input.cuda()
            heatmap_var = heatmap.cuda()

            self.optimizer.zero_grad()

            heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).cuda()
            vis = label[:, :, :, -1]
            vis = vis.view(-1, self.numClasses, 1)

            losses = {}
            loss   = 0

            start_model = time.time()
            heat = self.model(input_var, heatmap_var)
            losses = self.criterion(heat, heatmap_var)
            loss  += losses.item() #+ 0.5 * relation_loss.item()
            
            b, t, c, h, w = input.shape
           
            if self.is_visual:
                file_name = 'result/heats/{}_batch.jpg'.format(i)

                input = input.view(-1, c, h, w)
                heat = heat.view(-1, self.numClasses, heat.shape[-2], heat.shape[-1])
                
            input, heat = input.view(b, t, c, h, w).contiguous(), heat.view(b, t, self.numClasses, heat.shape[-2], heat.shape[-1]).contiguous()

            for j in range(heat.size(0)): #self.frame_memory):
                acc, acc_PCK, acc_PCKh, cnt, pred, visible = evaluate.accuracy(heat[j].detach().cpu().numpy(),\
                                            heatmap_var[j].detach().cpu().numpy(), pck_thred, 0.5, self.dataset, bbox[j], normTorso=True)
                preds.append(pred)
                for k in range(self.numClasses):
                    if visible[k] == 1:
                        if k != 0 or k != 1:
                            AP[k] = (AP[k] * count[k] + acc[k]) / (count[k]+1)
                            PCK[k] = (PCK[k] * count[k] + acc_PCK[k]) / (count[k]+1)
                            PCKh[k] = (PCKh[k]* count[k] + acc_PCKh[k]) / (count[k]+1)
                            count[k] += 1
                        
            mAP = AP[:].sum()/(self.numClasses)
            mPCK = PCK[:].sum()/(self.numClasses)
            mPCKh = PCKh[:].sum()/(self.numClasses)


            val_loss += loss

            tbar.set_postfix(valoss='%.6f' % (val_loss / self.batch_size), mPCK=mPCK)

        printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, self.dataset)
        
        res_pck[:-1] = PCK
        res_pck[-1] = mPCK
        np.save('result/hmdb_pck/PCK@{}.npy'.format(pck_thred), res_pck)

        PCKhAvg = PCKh.sum()/(self.numClasses)
        PCKAvg  =  PCK.sum()/(self.numClasses)
        times = time.strftime('%Y%m%d', time.localtime())

        if mPCK > self.isBest:
            self.isBest = mPCK
            source_model_save_path = self.args.source_model_save_path + self.args.model_name
            if not os.path.exists(source_model_save_path):
                os.makedirs(source_model_save_path)
            if self.is_train is True:
                shutil.copy2('lib/models/dkd_net.py', source_model_save_path)
                np.save('experiments/best_index', start_index)
                save_checkpoint({'state_dict': self.model.state_dict()}, self.isBest, self.args.model_name+'_'+times, self.args.model_save_path)

        if mPCKh > self.bestPCKh:
            self.bestPCKh = mPCKh
        if mPCK > self.bestPCK:
            self.bestPCK = mPCK
            self.best_epoch = epoch

        print("Best epoch: %d; PCK = %2.2f%%; PCKh = %2.2f%%" % (self.best_epoch, self.bestPCK*100,self.bestPCKh*100))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None, type=str, dest='pretrained')
    parser.add_argument('--dataset', type=str, dest='dataset', default='Sub-JHMDB')
    parser.add_argument('--train_dir', default=None,type=str, dest='train_dir')
    parser.add_argument('--val_dir', type=str, dest='val_dir', default=None)
    parser.add_argument('--model_name', default='SPSTN-Res-50', type=str)
    parser.add_argument('--model_save_path', default='experiments/checkpoint/')
    parser.add_argument('--source_model_save_path', default='experiments/source-model/')
    parser.add_argument('--model_arch', default=None, type=str)
    parser.add_argument('--writer', default=None)
    parser.add_argument('--is_train', default=False, type=bool)
    parser.add_argument('--visual', default=False, type=bool, help='If visualize results')
    
    args = parser.parse_args()
    
    starter_epoch = 0
    epochs =  100
    
    is_train = args.is_train
    is_visual = args.visual
    args.dataset    = 'Sub-JHMDB'
    print(is_train)
    args.frame_memory = 5
    
    if args.dataset == 'Sub-JHMDB':
        args.train_dir  = 'data/Sub-JHMDB/'
        args.val_dir    = 'data/Sub-JHMDB/'

    if is_train:
        trainer = Trainer(args, is_train=True, is_visual=False)
        for epoch in range(starter_epoch, epochs):
            trainer.training(epoch)
            trainer.test(epoch)
    else:
        print('Testing')
        trainer = Trainer(args, is_train=False, is_visual=True)
        trainer.test(0)
