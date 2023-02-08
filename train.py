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
# ITES, RPSTN 상대경로 지정
import os
os.environ[‘KMP_DUPLICATE_LIB_OK’]=True
from ITES import train_student
from RPSTN.pose_estimation import train_penn
from joint_heatmap import generate_2d_integral_preds_tensor
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
        self.gpus = [int(i) for i in config.GPUS.split(',')]
        self.is_train = is_train
        self.is_visual = is_visual
        ## JRE
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
        ## ITES
        self.num_joints = 17
        self.n_fully_connected = 1024
        self.n_layers = 6
        self.basis = 12
        self.init_std = 0.01


        if self.dataset ==  "Penn_Action":
            self.numClasses = 13
            self.test_dir = None
        def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
            out_poses_3d = []
            out_poses_2d = []
            out_camera_params = []
            for subject in subjects:
                for action in keypoints[subject].keys():
                    if action_filter is not None:
                        found = False
                        for a in action_filter:
                            if action.startswith(a):
                                found = True
                                break
                        if not found:
                            continue

                    poses_2d = keypoints[subject][action]
                    for i in range(len(poses_2d)):  # Iterate across cameras
                        out_poses_2d.append(poses_2d[i])

                    if subject in dataset.cameras():
                        cams = dataset.cameras()[subject]
                        assert len(cams) == len(poses_2d), 'Camera count mismatch'
                        for i,cam in enumerate(cams):
                            if 'intrinsic' in cam:
                                out_camera_params.append(np.tile((cam['intrinsic'])[None,:],(len(poses_2d[i]),1)))

                    if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                        poses_3d = dataset[subject][action]['positions_3d']
                        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                        for i in range(len(poses_3d)):  # Iterate across cameras
                            out_poses_3d.append(poses_3d[i])

            if len(out_camera_params) == 0:
                out_camera_params = None
            if len(out_poses_3d) == 0:
                out_poses_3d = None

            stride = args.downsample
            if subset < 1:
                for i in range(len(out_poses_2d)):
                    n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
                    start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                    out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
                    if out_poses_3d is not None:
                        out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
            elif stride > 1:
                # Downsample as requested
                for i in range(len(out_poses_2d)):
                    out_poses_2d[i] = out_poses_2d[i][::stride]
                    if out_poses_3d is not None:
                        out_poses_3d[i] = out_poses_3d[i][::stride]
                        out_camera_params[i] = out_camera_params[i][::stride]

            return out_camera_params, out_poses_3d, out_poses_2d
        self.train_loader, self.val_loader, self.test_loader = getDataloader(self.dataset, self.train_dir, \
                                                                self.val_dir, self.test_dir, self.sigma, self.stride, \
                                                                self.workers, self.frame_memory, \
                                                                self.batch_size)
        #loader output = images, label_map, label, img_paths, person_box, start_index,kpts
        model_jre = train_penn.models.dkd_net.get_dkd_net(config, self.is_visual, is_train=True if self.is_train else False)

        model_pos_train = train_student.Teacher_net(self.num_joints,self.num_joints,2,  # joints = [13,2]
                            n_fully_connected=self.n_fully_connected, n_layers=self.n_layers, 
                            dict_basis_size=self.basis, weight_init_std = self.init_std)
        self.model_jre = torch.nn.DataParallel(model, device_ids=self.gpus).cuda()
        self.criterion_jre = MSESequenceLoss().cuda()
        self.optimizer_jre = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_ite = torch.optim.SGD(model_pos_train.parameters(), lr=lr,
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
            learning_rate = adjust_learning_rate(self.optimizer, epoch, self.lr, weight_decay=self.weight_decay, policy='multi_step',
                                                 gamma=self.gamma, step_size=self.step_size)

            vis = label[:, :, :, -1]
            vis = vis.view(-1, self.numClasses, 1)

            input_var = input.cuda()
            heatmap_var = heatmap.cuda()
            kpts = kpts[:13] # joint
            heat = torch.zeros(self.numClasses, self.heatmap_size, self.heatmap_size).cuda()

            losses = {}
            loss = 0
            start_model = time.time()
            heat = self.model(input_var)
            losses = self.criterion(heat, heatmap_var)
            jfh  = generate_2d_integral_preds_tensor(heat , self.num_joints, self.heatmap_size,self.heatmap_size) # joint from heatmap K , 64 , 64 
            preds = model_pos_train(jfh,align_to_root=True)
            
            loss_total =  loss_reprojection + loss_consistancy
            loss_reprojection = preds['l_reprojection'] 
            loss_consistancy = preds['l_cycle_consistent']
            loss_total.backward()

            optimizer_ite.step()
            # output => [ba , num_joints , 2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_jre', default=None, type=str, dest='pretrained')