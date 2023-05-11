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


dataset_path = 'data/data_3d_' + 'h36m'+ '.npz'
dataset = Human36mDataset(dataset_path)
def mask_joint(joint,mlm_probability=0.2,pair = True): # ba, joint , 2 , Pair 를 동시에 제거
    m = torch.full(joint.shape,mlm_probability) # 40 , 16 , 2
    if pair: 
        masked_indices = torch.bernoulli(m[:,:,0]).bool() # batch , 17
        # cp = torch.repeat(joint.shape[0],masked_indices[-1]) # 40 , 16 
        masked_indices = torch.stack([masked_indices,masked_indices],dim = 2)
    else:
        masked_indices = torch.bernoulli(m).bool()
    m[masked_indices] = 1e-9
    m[~masked_indices] = 1
    m = m.cuda()
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
    scale = (1/c) / mean_bone
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
    ind = torch.tensor([13,7,9,11,8,10,12,14,15,0,16,2,4,6,1,3,5]).cuda()
    #[9,14,11,15,12,16,13,1,4,2,5,3,6,0,7,8,10]
    #[9,14,11,15,12,16,13,1,4,2,5,6,3,0,7,8,10]
    #[10,14,11,15,12,16,13,1,4,2,5,3,6,0,7,8,10]
    #[9,14,11,15,12,16,13,1,4,2,5,3,6,0,7,8,10]
    jfh = torch.index_select(jfh, dim=1, index=ind)
    return jfh

class Trainer(object):
    def __init__(self, args, is_train, is_visual):
        self.args = args
        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.model_arch = args.model_arch
        self.dataset = args.dataset
        self.frame_memory = args.frame_memory   
        self.writer = args.writer
        self.gpus =  [int(i) for i in train_penn.config.GPUS.split(',')]
        self.is_train = is_train
        self.is_visual = is_visual
        ## JRE
        self.writer = SummaryWriter('exp/tensor/3d')
        self.test_dir = None
        self.workers = 1
        self.weight_decay = 0.1
        self.momentum = 0.9
        self.batch_size = 2
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
        self.n_layers = 6
        self.basis = 12
        self.init_std = 0.01
        self.hid_dim = 128
        self.n_blocks = 4
        adj = adj_mx_from_skeleton(dataset.skeleton())
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
                            dict_basis_size=self.basis, weight_init_std = self.init_std).cuda()
        self.model_jre = torch.nn.DataParallel(model_jre, device_ids=self.gpus).cuda()
        self.submodel = Student_net(adj, self.hid_dim, num_layers=self.n_blocks, p_dropout=0.0,
                       nodes_group=dataset.skeleton().joints_group()).cuda()
        if args.pretrained:
            self.model_jre.load_state_dict(torch.load(args.pretrained)['state_dict'])
            checkpoint = torch.load('ITES/checkpoint/teacher/ckpt_teacher.bin')#, map_location=lambda storage, loc: storage)
            self.model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
        self.criterion_jre = train_penn.MSESequenceLoss().cuda()
        if args.sub_trained:
            self.submodel.load_state_dict(torch.load('exp/submodel/tea_model_epoch_82.bin')['model_pos'],strict = False)
        if args.pretrained:
            self.param = list(self.model_pos_train.parameters())
        else:
            self.param = list(self.model_jre.parameters()) + list(self.model_pos_train.parameters())
        self.optimizer = torch.optim.SGD( self.model_pos_train.parameters(), lr=0.001,
                            momentum=0.9,
                            weight_decay=0.0005)
        if args.submodule:
            self.param = list(self.submodel.parameters())
            self.sub_optimizer = torch.optim.AdamW(self.submodel.parameters(), lr=0.001,
                            weight_decay=0.0005)
        #self.optimizer = torch.optim.Adam(self.param, lr=self.lr)

  #      self.optimizer_ite = torch.optim.SGD(self.model_pos_train.parameters(), lr=self.lr,
  #                          momentum=args.momentum,
  #                          weight_decay=args.weight_decay)

        self.iters = 0
        pretrained_jre = None

        self.isBest = 0
        self.bestPCK  = 0
        self.bestPCKh = 0
        self.best_epoch = 0


    def training(self, epoch):
        print('Start Training....')
        train_loss = 0.0
        self.model_jre.train()
        self.model_pos_train.train()
        optimizer = self.optimizer
        args = self.args
        if args.submodule:
            sub_optim = self.sub_optimizer
            self.submodel.train()
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)
        t_loss =0
        for i, (input, heatmap, label, img_path, bbox, start_index, kpts) in enumerate(tbar):
            learning_rate = train_penn.adjust_learning_rate(self.optimizer, epoch, self.lr, weight_decay=self.weight_decay, policy='multi_step',
                                                 gamma=self.gamma, step_size=self.step_size)
            optimizer.zero_grad()
            vis = label[:, :, :, -1]
            vis = vis.view(-1, self.numClasses, 1)  
            input_var = input.cuda()
            heatmap_var = heatmap.cuda()
            heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).cuda()
            heat = self.model_jre(input_var)
            # self.iters += 1
            #[8, 5, 16, 64, 64]
            kpts = kpts[:13]
            kpts = kpts.reshape(-1,13,2)
            jfh  = generate_2d_integral_preds_tensor(heat , 13, self.heatmap_size,self.heatmap_size)
            losses = {}
            loss = 0
            losses = self.criterion_jre(heat, heatmap_var)
            loss += losses
            jre_loss = loss.item()
            # joint from heatmap K , 64 , 64  [40, 13, 2]
            jfh_copy = jfh
            jfh = make_joint(jfh)
            jfh = jfh.cuda()
            jfh = normalize_2d(jfh)
            kpts = kpts.cuda()
            kpts = make_joint(kpts)
            kpts = normalize_2d(kpts)
            kpts = kpts.type(torch.float).cuda()
            if args.submodule:
                sub_optim.zero_grad()
                kpts_mask = mask_joint(kpts)
                preds = self.submodel(kpts_mask)
                reconstruct = preds['reconstruct']
                train_loss = self.criterion_jre(kpts,reconstruct)
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
            if args.submodule:
                sub_optim.step()
            else:
                optimizer.step()

            #self.writer.add_scalar('jre_loss', (losses / self.batch_size), epoch)
            #self.writer.add_scalar('total_loss', (loss_total / self.batch_size), epoch)
            path = f'exp/train/skeleton2d/{epoch}.jpg'
            if self.is_visual == True and i == 0:
                if epoch % 1 == 0 :
                    b, t, c, h, w = input.shape
                    file_name = 'result/heats/train/{}_epoch.jpg'.format(epoch)
                    input = input.view(-1, c, h, w)
                    heat = heat.view(-1, 13, heat.shape[-2], heat.shape[-1])
                    train_penn.save_batch_heatmaps(path,input,heat,file_name,jfh_copy)
            with torch.no_grad():
                if args.submodule:
                    vis_joint = preds['reconstruct']
                else:
                    vis_joint = preds['shape_camera_coord']
                    vis_joint2 = preds_1['reconstruct']
                # preds['shape_camera_coord'] <- 2차원 projection 좌표계
                # 2차원 사진 가져오기
                vis_joint = vis_joint.cpu()
                # np.save('3dpred.npy',vis_joint.numpy())
                if epoch % 1 == 0 :
                    if i  == 0:
                        for j in range(3):
                            sub_path = f'exp/img/train/{epoch}_{j}.jpg'
                            image = input[j].mul(255)\
                        .clamp(0, 255)\
                        .byte()\
                        .permute(1, 2, 0)\
                        .cpu().numpy()
                            draw_3d_pose1(vis_joint[i],dataset.skeleton(),'visualization_custom/' + 'train/'+str(epoch) + '_' +str(j)+'_teacher_result.jpg')
                            draw_2d_pose(vis_joint[i],dataset.skeleton(),'visualization_custom/' + '2dtrain_notsub/'+str(epoch) + '_' +str(j)+'_teacher_result.jpg')
                            draw_2d_pose(vis_joint2[i],dataset.skeleton(),'visualization_custom/' + '2dtrain_notsub_submodule/'+str(epoch) + '_' +str(j)+'_teacher_result.jpg')

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
        if args.submodule:
            self.submodel.eval()
        with torch.no_grad():
            for i, (input, heatmap, label, img_path, bbox, start_index,kpts) in enumerate(tbar):
                cnt += 1
                idx.append(start_index)
                input_var = input.cuda()
                heatmap_var = heatmap.cuda()
                kpts = kpts[:13]
                self.optimizer.zero_grad()

                heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).cuda()
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
                jfh = jfh.cuda()
                kpts = kpts.cuda()
                kpts = make_joint(kpts)
                kpts = normalize_2d(kpts)
                jfh  = make_joint(jfh)
                jfh = normalize_2d(jfh)
                kpts = kpts.type(torch.float).cuda()
                #permute = [10,14,11,15,12,16,13,1,4,2,5,3,6,0,7,8,10]
                if args.submodule:
                    kpts_mask = mask_joint(kpts)
                    preds = self.submodel(kpts_mask)
                    reconstruct = preds['reconstruct']
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
                    val_loss = loss_total + losses
                    #[8,5,3,256,256]?
                b, t, c, h, w = input.shape
            
                #if self.is_visual:
                file_name = 'result/heats/val/{}_epoch.jpg'.format(epoch)
                path = f'exp/val/skeleton2d/{epoch}.jpg'
                input = input.view(-1, c, h, w)
                if epoch % 5 == 0 and i == 0 :
                    joint = generate_2d_integral_preds_tensor(heat , 13, self.heatmap_size,self.heatmap_size)
                    heat = heat.view(-1, 13, heat.shape[-2], heat.shape[-1])
                    train_penn.save_batch_heatmaps(path,input,heat,file_name,jfh)
                if epoch % 1 == 0 :
                    if args.submodule:
                        vis_joint = preds['reconstruct']
                    else:
                        vis_joint = preds['shape_camera_coord']
                        vis_joint2 = preds_1['reconstruct']
                    vis_joint = vis_joint.cpu()
                    if i == 0:
                        for j in range(1):
                            sub_path = f'exp/img/test/{epoch}_{j}.jpg'
                            image = input[j].mul(255)\
                        .clamp(0, 255)\
                        .byte()\
                        .permute(1, 2, 0)\
                        .cpu().numpy()
                            draw_3d_pose1(vis_joint[i],dataset.skeleton(),'visualization_custom/'+'test/'+str(epoch) + '_'+str(j)+'val_teacher_result.jpg')
                            draw_2d_pose(vis_joint[i],dataset.skeleton(),'visualization_custom/' + '2dtest_notsub/'+str(epoch) + '_' +str(j)+'_teacher_result.jpg')
                            draw_2d_pose(vis_joint2[i],dataset.skeleton(),'visualization_custom/' + '2dtest_notsub_submodule/'+str(epoch) + '_' +str(j)+'_teacher_result.jpg')
        self.writer.add_scalar('val_loss', (val_loss/ self.batch_size), epoch)
        if epoch >= 1:
            chk_path= os.path.join(args.checkpoint, 'tea_model_epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            if args.submodule:
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
    parser.add_argument('--submodule' , default = False,type=bool)
    parser.add_argument('--sub_trained',default = False , type = str  )
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
        args.train_dir  = '../data/pose_data/itedata'
        args.val_dir    = '../data/pose_data/itedata'
        tb_log_dir = 'run/'
    else:
        args.train_dir  = '../data/pose_data'
        args.val_dir    = '../data/pose_data'
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
