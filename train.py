import sys
sys.path.append("ITES")
sys.path.append("ITES/common")
sys.path.append("ITES/data")
sys.path.append("ITES/img")
sys.path.append("ITES/checkpoint")
sys.path.append('RPSTN/custom')
sys.path.append('RPSTN/files')
sys.path.append('RPSTN/lib')
sys.path.append('RPSTN/pose_estimation')
sys.path.append('RPSTN/lib/utils')
# ITES, RPSTN 상대경로 지정 python RPSTN/pose_estimation/train_penn.py
import os
import pdb
from ITES.common.utils import deterministic_random
#os.environ["KMP_DUPLICATE_LIB_OK"] = True
from ITES import train_t
from RPSTN.pose_estimation import train_penn
from joint_heatmap import generate_2d_integral_preds_tensor
from RPSTN.lib.utils import evaluate as evaluate
from tensorboardX import SummaryWriter
import argparse
import torch
from tqdm import tqdm
import time
import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class Trainer(object):
    def __init__(self, args, is_train, is_visual):
        self.args = args
        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.model_arch = args.model_arch
        self.dataset = args.dataset
        self.frame_memory = args.frame_memory   
        self.writer = args.writer
        self.gpus = 1#[int(i) for i in config.GPUS.split(',')]
        self.is_train = is_train
        self.is_visual = is_visual
        ## JRE
        self.writer = SummaryWriter(args.dir)
        self.test_dir = None
        self.workers = 1
        self.weight_decay = 0.1
        self.momentum = 0.9
        self.batch_size = 8
        self.lr = 0.0005
        self.gamma = 0.333
        self.step_size = [8, 15, 25, 40, 80]#13275
        self.sigma = 2
        self.stride = 4
        self.heatmap_size = 64
        self.is_visual = True
        ## ITES
        self.num_joints = 16
        self.n_fully_connected = 1024
        self.n_layers = 6
        self.basis = 12
        self.init_std = 0.01


        if self.dataset ==  "Penn_Action":
            self.numClasses = 16
            self.test_dir = None
        self.train_loader, self.val_loader, self.test_loader = train_penn.getDataloader(self.dataset, self.train_dir, \
                                                                self.val_dir, self.test_dir, self.sigma, self.stride, \
                                                                self.workers, self.frame_memory, \
                                                                self.batch_size)
        #loader output = images, label_map, label, img_paths, person_box, start_index,kpts
        model_jre = train_penn.models.dkd_net.get_dkd_net(train_penn.config, self.is_visual, is_train=True if self.is_train else False)

        self.model_pos_train = train_t.Teacher_net(self.num_joints,self.num_joints,2,  # joints = [13,2]
                            n_fully_connected=self.n_fully_connected, n_layers=self.n_layers, 
                            dict_basis_size=self.basis, weight_init_std = self.init_std)
        self.model_jre = torch.nn.DataParallel(model_jre, device_ids=self.gpus).cuda()
        self.criterion_jre = train_penn.MSESequenceLoss().cuda()
        self.optimizer_jre = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_ite = torch.optim.SGD(self.model_pos_train.parameters(), lr=self.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

        self.iters = 0

        if self.args.pretrained_jre is not None:
            checkpoint = torch.load(self.args.pretrained)
            p = checkpoint['state_dict']
            if self.dataset == "Penn_Action":
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
        for i, (input, heatmap, label, img_path, bbox, start_index, kpts) in enumerate(tbar):
            learning_rate = train_penn.adjust_learning_rate(self.optimizer, epoch, self.lr, weight_decay=self.weight_decay, policy='multi_step',
                                                 gamma=self.gamma, step_size=self.step_size)

            vis = label[:, :, :, -1]
            vis = vis.view(-1, self.numClasses, 1)

            input_var = input.cuda()
            heatmap_var = heatmap.cuda()
            # self.iters += 1
            self.writer.add_scalar('train_loss', (train_loss / self.batch_size), epoch)
            if self.is_visual == True:
                if epoch % 5 == 0 :
                    b, t, c, h, w = input.shape
                    file_name = 'result/heats/train/{}_batch.jpg'.format(epoch)
                    input = input.view(-1, c, h, w)
                    heat = heat.view(-1, 16, heat.shape[-2], heat.shape[-1])
                    train_penn.save_batch_heatmaps(input,heat,file_name)

            kpts = kpts[:16] # joint
            heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).cuda()

            losses = {}
            loss = 0
            start_model = time.time()
            heat = self.model(input_var)
            losses = self.criterion(heat, heatmap_var)
            jfh  = generate_2d_integral_preds_tensor(heat , self.num_joints, self.heatmap_size,self.heatmap_size) # joint from heatmap K , 64 , 64 
            preds = self.model_pos_train(jfh,align_to_root=True)
            # Batch, 16,2
            loss_total =  loss_reprojection + loss_consistancy
            loss_reprojection = preds['l_reprojection'] 
            loss_consistancy = preds['l_cycle_consistent']
            train_loss = loss_total + losses
            self.writer.add_scalar('train_loss', (train_loss / self.batch_size), epoch)
            loss_total.backward()
            
            optimizer_ite.step()
            # output => [ba , num_joints , 2]
    def validation(self, epoch):
        print('Start Testing....')
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        
        AP = np.zeros(self.numClasses)
        PCK = np.zeros(self.numClasses)
        PCKh = np.zeros(self.numClasses)
        count = np.zeros(self.numClasses)

        res_pck = np.zeros(self.numClasses + 1)
        pck_thred = 1

        idx = []
        cnt = 0
        preds = []

        for i, (input, heatmap, label, img_path, bbox, start_index,kpts) in enumerate(tbar):
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
            heat = self.model(input_var)
            losses = self.criterion(heat, heatmap_var)
            loss  += losses.item() #+ 0.5 * relation_loss.item()
            #[8,5,3,256,256]?
            b, t, c, h, w = input.shape
           
            #if self.is_visual:
            file_name = 'result/heats/{}_batch.jpg'.format(i)
            input = input.view(-1, c, h, w)
            pdb.set_trace()
            heat = heat.view(-1, 16, heat.shape[-2], heat.shape[-1])
            joint = generate_2d_integral_preds_tensor(heatmap_var , self.num_joints, self.heatmap_size,self.heatmap_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None, type=str, dest='pretrained')
    parser.add_argument('--dataset', type=str, dest='dataset', default='Penn_Action')
    parser.add_argument('--train_dir', default=None,type=str, dest='train_dir')
    parser.add_argument('--val_dir', type=str, dest='val_dir', default=None)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--model_save_path', default='experiments/checkpoint/')
    parser.add_argument('--source_model_save_path', default='experiments/source-model/')
    parser.add_argument('--model_arch', default=None, type=str)
    parser.add_argument('--writer', default=None)
    parser.add_argument('--is_train', default=False, type=bool)
    parser.add_argument('--visual', default=False, type=bool, help='If visualize results')
    parser.add_argument('--dir' , default = 'run',type=str)
    parser.add_argument('--pretrained_jre', default=None, type=str, dest='pretrained')
    RANDSEED = 2021
    starter_epoch = 0
    epochs =  100
    args = parser.parse_args()
    is_train = args.is_train
    is_visual = args.visual
    args.dataset  = 'pose_data'
    args.frame_memory = 5
    if args.dataset == 'pose_data':
        args.train_dir  = 'data/pose_data'
        args.val_dir    = 'data/pose_data'
        tb_log_dir = 'run/'
    args.train_dir  = 'data/pose_data'
    args.val_dir    = 'data/pose_data'
  #  tb_log_dir = 'run/penn/'
#    writer = SummaryWriter(log_dir= '', comment='weight_decay')
#    args.writer = writer
    set_seed(RANDSEED)
    if is_train == True:
        trainer = Trainer(args, is_train=True, is_visual=True) # 원래 False
        for epoch in range(starter_epoch, epochs):
            trainer.training(epoch)
            trainer.validation(epoch)
    else:
        trainer = Trainer(args, is_train=False, is_visual=True)
        trainer.validation(0)
