# python train.py --is_train True
import sys
sys.path.append("./")
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
from collections import OrderedDict
from ITES.common.utils import deterministic_random
#os.environ["KMP_DUPLICATE_LIB_OK"] = True
from ITES import train_t
from RPSTN.pose_estimation import train_penn
from conv_joint import generate_2d_integral_preds_tensor
from RPSTN.lib.utils import evaluate as evaluate
from tensorboardX import SummaryWriter
import argparse
import torch
from tqdm import tqdm
import time
import random
import numpy as np
from ITES.common.visualization import draw_3d_pose , draw_3d_pose1 , draw_2d_pose
from ITES.common.h36m_dataset import Human36mDataset
from ITES.common.function import *
from reconstruct_joint import Student_net



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_path = 'ITES/data/data_3d_' + 'h36m'+ '.npz'
dataset = Human36mDataset(dataset_path)
def mask_joint(joint,mlm_probability=0.2,pair = True): # ba, joint , 2 , Pair 를 동시에 제거
    m = torch.full(joint.shape,mlm_probability) # 40 , 16 , 2
    if pair: 
        masked_indices = torch.bernoulli(m[:,:,0]).bool() # batch , 17
        # cp = torch.repeat(joint.shape[0],masked_indices[-1]) # 40 , 16 
        masked_indices = torch.stack([masked_indices,masked_indices],dim = 2)
    else:
        masked_indices = torch.bernoulli(m).bool()
    m[masked_indices] = 1e-5
    m[~masked_indices] = 1
    m = m.to('cuda')
    m_joint = joint * m 
    return m_joint # masking 된 joint 값 출력

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_2d(pose):
    # pose:(N,J,2)
    mean_bone = torch.mean(torch.linalg.norm(pose[:,0:1,:]-pose[:,10:11,:],axis=2,ord=2)) #hip to head
    c = 5
    scale = (1/c) / (mean_bone + 1e-8)
    pose = pose * scale
    return pose 
# 만약 전체가 나오지 않는다면?
def make_joint(jfh):
    rev = (jfh[:,7] + jfh[:,8])/2
    rev = rev.reshape(-1,1,2)
    spine = (jfh[:,7]+jfh[:,8]+jfh[:,1]+jfh[:,2])/4
    spine = spine.reshape(-1,1,2)
    neck = (jfh[:,0] + jfh[:,1] + jfh[:,2])/3
    neck = neck.reshape(-1,1,2)
    top = jfh[:,0].reshape(-1,1,2)
    jfh = torch.cat([jfh,rev],dim = 1)
    jfh = torch.cat([jfh,spine],dim = 1)
    jfh = torch.cat([jfh,neck],dim = 1)
    jfh = torch.cat([jfh,top],dim = 1)
    ind = torch.tensor([13,7,9,11,8,10,12,14,15,0,16,2,4,6,1,3,5]).to('cuda')
    jfh = torch.index_select(jfh, dim=1, index=ind)
    return jfh

class Trainer(object):
    def __init__(self, args, is_train, is_visual):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'
        self.args = args
        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.model_arch = args.model_arch
        self.dataset = args.dataset
        self.frame_memory = args.frame_memory   
        self.writer = args.writer
        self.gpus =  0
        self.is_train = is_train
        self.is_visual = is_visual
        ## JRE
        self.writer = SummaryWriter('exp/tensor/3d')
        self.test_dir = None
        self.workers = 4
        self.weight_decay = 0.1
        self.momentum = 0.9
        self.batch_size = args.batch
        self.lr = 0.0005
        self.gamma = 0.333
        self.step_size = [8, 15, 25, 40, 80]#13275
        self.sigma = 2
        self.stride = 4
        self.heatmap_size = 64
        self.is_visual = True
        self.ground = args.ground  # 시각화에 GT를 사용할지의 여부
        ## ITES
        self.num_joints = 17
        self.n_fully_connected = 1024
        self.n_layers = 4
        self.basis = 12
        self.init_std = 0.01
        self.hid_dim = 128
        self.n_blocks = 4
        adj = adj_mx_from_skeleton(dataset.skeleton())
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
        if self.dataset ==  "pose_data":
            self.numClasses = 13
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
        self.model_jre = torch.nn.DataParallel(model_jre)
        self.model_jre = self.model_jre.to('cuda')
        loaded_state_dict = torch.load('exp/checkpoints/penn_train_20230624_best.pth.tar')['state_dict']
        self.submodel = Student_net(adj, self.hid_dim, num_layers=self.n_blocks, p_dropout=0.0,
                       nodes_group=dataset.skeleton().joints_group())
        self.submodel = torch.nn.DataParallel(self.submodel)
        self.submodel = self.submodel.to('cuda')
        self.model_jre.load_state_dict(loaded_state_dict)
        if args.pretrained:
            #self.model_jre.load_state_dict(torch.load(args.pretrained)['state_dict'])
            checkpoint = torch.load('ITES/checkpoint/teacher/ckpt_teacher.bin')#, map_location=lambda storage, loc: storage)
            self.model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
        self.criterion_jre = train_penn.MSESequenceLoss().to('cuda')
        if args.sub_trained:
            self.submodel.load_state_dict(torch.load('exp/checkpoints/submodule/best.bin')['model_pos'],strict = False)
        if args.pretrained:
            self.param = list(self.model_pos_train.parameters())
        else:
            self.param = list(self.model_jre.parameters()) + list(self.model_pos_train.parameters())
        self.optimizer = torch.optim.AdamW( self.model_pos_train.parameters(), lr=0.001,
                            weight_decay=0.0005)
        if args.submodule:
            self.param = list(self.submodel.parameters())
            self.sub_optimizer =  torch.optim.AdamW(self.submodel.parameters(), lr=0.001,
                            weight_decay=0.0005)

        self.iters = 0
        pretrained_jre = None

        self.isBest = 0
        self.bestPCK  = 0
        self.bestPCKh = 0   
        self.best_epoch = 0


    def training(self, epoch):
        print('Start Training....0801')
        train_loss = 0.0
        self.model_jre.train()
        self.model_pos_train.train()
        optimizer = self.optimizer
        args = self.args
        self.criterion_jre = self.criterion_jre.to('cuda')
        if args.submodule:
            sub_optim = self.sub_optimizer
            self.submodel.train()
            self.model_jre.eval()
            self.model_pos_train.eval()
        if args.sub_trained:
            self.submodel.eval()
        print("Epoch " + str(epoch) + ':') 
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        tbar = tqdm(self.train_loader)
        t_loss =0
        for i, (input, heatmap, label, img_path, bbox, start_index, kpts) in enumerate(tbar):
           # optimizer.zero_grad()
            sub_optim.zero_grad()
            vis = label[:, :, :, -1]
            
            vis = vis.view(-1, self.numClasses, 1)  
            input_var = input.to('cuda')
            heatmap_var = heatmap.to('cuda')
            heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).to('cuda')
            heat = self.model_jre(input_var).to('cuda')
            # self.it   ers += 1
            #[8, 5, 16, 64, 64]
            kpts = kpts[:13]
            kpts = kpts.reshape(-1,13,2)
            jfh  = generate_2d_integral_preds_tensor(heat , 13, self.heatmap_size,self.heatmap_size)
            losses = {}
            loss = 0
            losses = self.criterion_jre(heat, heatmap_var).to('cuda')
            loss += losses
            jre_loss = loss.item()
            # joint from heatmap K , 64 , 64  [40, 13, 2]
            jfh_copy = jfh
            jfh = make_joint(jfh)
            jfh = jfh.to('cuda')
            jfh = normalize_2d(jfh)
            kpts = kpts.to('cuda')
            kpts = make_joint(kpts)
            kpts = normalize_2d(kpts)
            kpts = kpts.type(torch.float).to('cuda')
            # 디버거 걸어서 봐보자. <- if 문 걸어서 inf 발생 했다면 브레이크 해서 직전에 텐서 모양 확인하보기.
            # 모델이 복사가 되는 거고 직렬적으로 되는 것이 아니다, 학습 시킬 때 속도의 차이가 있는 것이다.
            # Invisible Joint 에 대해서 찾아보도록 하자.
            # Supervised 에서도 있는지 한번 찾아보자.
            if args.submodule:
                kpts_mask = mask_joint(kpts) # 한번더 학습 시키기
                preds = self.submodel(kpts_mask)
                reconstruct = preds['reconstruct']
                reconstruct = reconstruct.to('cuda')
                train_loss = self.criterion_jre(kpts,reconstruct)
                print("Outside: input size", kpts_mask.size(),
                "output_size", reconstruct.size())
                torch.autograd.set_detect_anomaly(True)
            # 
            else:
                #jfh_mask = mask_joint(jfh)
                preds_1 = self.submodel(jfh)
                joint = preds_1['reconstruct']
                preds = self.model_pos_train(joint,align_to_root=True)
                #pdb.set_trace()
                # Batch, 16,2          
                loss_reprojection = preds['l_reprojection'] 
                loss_consistancy = preds['l_cycle_consistent']
                loss_total =  loss_reprojection + loss_consistancy
                #if args.pretrained:
                train_loss = loss_total # + jre_loss
                #else:
                #    train_loss = loss_total + jre_loss
            train_loss.backward()
            t_loss += train_loss
            sub_optim.step()
            #if args.submodule:
            #    sub_optim.step()
            #else:
            #    optimizer.step()

            #self.writer.add_scalar('jre_loss', (losses / self.batch_size), epoch)
            #self.writer.add_scalar('total_loss', (loss_total / self.batch_size), epoch)

        self.writer.add_scalar('teacher_loss', (t_loss / self.batch_size), epoch)
#        with torch.no_grad():
#            vis_joint = preds['shape_camera_coord']
#            if epoch % 5 == 0 :
#                for i in range(10):
#                    draw_3d_pose(vis_joint[i,:,:],f'exp/vis/{epoch}_{i}.jpg')

            # output => [ba , num_joints , 2]
    def validation(self, epoch):
        print('Start Testing....')
        model_jre = self.model_jre
        model_ite = self.model_pos_train
        tbar = tqdm(self.val_loader, desc='\r')
        args = self.args
        val_loss = 0.0
        model_jre.eval()
        model_ite.eval()
        self.submodel.eval()
        AP = np.zeros(self.numClasses)
        PCK = np.zeros(self.numClasses)
        PCKh = np.zeros(self.numClasses)
        count = np.zeros(self.numClasses)

        res_pck = np.zeros(self.numClasses + 1)
        pck_thred = 1

        idx = []
        cnt = 0
        vt_loss = 0
        preds = []
        with torch.no_grad():
            for i, (input, heatmap, label, img_path, bbox, start_index,kpts) in enumerate(tbar):
                cnt += 1
                idx.append(start_index)
                input_var = input.to('cuda')
                heatmap_var = heatmap.to('cuda')
                kpts = kpts[:13]

                heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).to('cuda')
                vis = label[:, :, :, -1]
                vis = vis.view(-1, self.numClasses, 1)
                heat = model_jre(input_var)
                losses = {}
                loss = 0
                kpts = kpts[:13]   
                kpts = kpts.reshape(-1,13,2)
                # joint from heatmap K , 64 , 64 
                jfh  = generate_2d_integral_preds_tensor(heat , 13, self.heatmap_size,self.heatmap_size)
                jfh  = generate_2d_integral_preds_tensor(heatmap_var , 13, self.heatmap_size,self.heatmap_size)
                jfh = jfh.to('cuda')
                kpts = kpts.to('cuda') # 64X64
                kpts = make_joint(kpts)
                kpts = normalize_2d(kpts)
                jfh  = make_joint(jfh)
                jfh = normalize_2d(jfh)
              #   kpts = kpts.type(torch.float).cuda()
                #permute = [10,14,11,15,12,16,13,1,4,2,5,3,6,0,7,8,10]
                if args.submodule:
                    kpts_mask = mask_joint(jfh) # 한번더 학습시키기
                    preds = self.submodel(kpts_mask)
                    reconstruct = preds['reconstruct']
                    reconstruct = reconstruct.to('cuda')
                    val_loss += self.criterion_jre(kpts,reconstruct)
                else:
                    #jfh_mask = mask_joint(jfh)
                    preds_1 = self.submodel(jfh)
                    joint = preds_1['reconstruct']
                    preds = self.model_pos_train(joint,align_to_root=True)
                    # Batch, 13,2
                    loss_reprojection = preds['l_reprojection'] 
                    loss_consistancy = preds['l_cycle_consistent']
                    loss_total =  loss_reprojection + loss_consistancy
                    heat = model_jre(input_var)
                    losses = self.criterion_jre(heat, heatmap_var)
                # loss  += losses.item() #+ 0.5 * relation_loss.item()
                    val_loss = loss_total #+ losses
                    #[8,5,3,256,256]?
                b, t, c, h, w = input.shape

        self.writer.add_scalar('val_loss', (val_loss/ self.batch_size), epoch)
        #shape_camera_coord = preds['shape_camera_coord']
        #depth = shape_camera_coord[:,:,2:3]

        if epoch >= 1:
            chk_path= os.path.join(args.checkpoint, 'tea_model_epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            if args.submodule:
                chk_path= os.path.join(args.checkpoint, 'submodel/submodel_{}.bin'.format(epoch))
                torch.save({
                'epoch': epoch,
                'lr': self.lr,
                'optimizer': self.sub_optimizer.state_dict(),
                'model_pos':self.submodel.state_dict(),
            }, chk_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'lr': self.lr,
                    'optimizer': self.optimizer.state_dict(),
                    'model_pos':self.model_pos_train.state_dict(),
                }, chk_path)

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
    parser.add_argument('--ground' , default = False,type=bool)
    parser.add_argument('--checkpoint' , default = 'exp/3d_ckpt',type=str)
    parser.add_argument('--submodule' , default = True,type=bool)
    parser.add_argument('--sub_trained',default = False , type = bool )
    parser.add_argument('--batch',default = 8 , type = int )
   # parser.add_argument('--pretrained_jre', default=None, type=str)
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
    else:
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
