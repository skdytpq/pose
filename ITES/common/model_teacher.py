import torch
import torch.nn as nn
from common.function import so3_exponential_map
from common.camera import *
from common.function import *
from common.loss import *
import pdb
class Teacher_net(nn.Module):
    def __init__(self, num_joints_in, num_joints_out, in_features=2, n_fully_connected=1024, n_layers=6, dict_basis_size=12, weight_init_std = 0.01):
        super().__init__()
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.n_fully_connected = n_fully_connected
        self.n_layers = n_layers
        self.dict_basis_size = dict_basis_size

        self.fe_net = nn.Sequential(
            *self.make_trunk(dim_in=self.num_joints_in * 2,
                             n_fully_connected=self.n_fully_connected,
                             n_layers=self.n_layers)) # Convolution Batchnormailization fully connected layer
        
        self.alpha_layer = conv1x1(self.n_fully_connected,
                            self.dict_basis_size,
                            std=weight_init_std)
        self.shape_layer = conv1x1(self.dict_basis_size, 3 * num_joints_in,
                                   std=weight_init_std) # 12 to 3 * num_joints
        self.rot_layer = conv1x1(self.n_fully_connected, 3,
                                 std=weight_init_std)
        self.trans_layer = conv1x1(self.n_fully_connected, 1,
                                 std=weight_init_std)
        self.cycle_consistent = True
        self.z_augment = False
        self.z_augment_angle = 0.2

    def make_trunk(self,
                   n_fully_connected=None,
                   dim_in=None,
                   n_layers=None,
                   use_bn=True):

        layer1 = ConvBNLayer(dim_in, # joint 의 개수를 1024의 channel 축으로 늘림
                             n_fully_connected,
                             use_bn=use_bn)
        layers = [layer1]

        for l in range(n_layers):
            layers.append(ResLayer(n_fully_connected,
                                   int(n_fully_connected/4)))

        return layers

    def forward(self, input_2d, align_to_root=False):
        assert input_2d.shape[1] == self.num_joints_in
        assert input_2d.shape[2] == self.in_features

        preds = {}
        ba = input_2d.shape[0] # batch
        dtype = input_2d.type()
        # pdb.set_trace()
        input_2d_norm, root = self.normalize_keypoints(input_2d) # Batch,num_joint,2 
        # root 는 각 Batch 의 머리 좌표값
        # Masking 을 하는 방법은 ? Visibility 사용 혹은 ?
        # Head 를 기준으로 정규화를 진행
        #pdb.set_trace()
        if self.z_augment: # random rotate 가 필요할 때 : consistency loss 계산
            R_rand = rand_rot(ba,
                              dtype=dtype,
                              max_rot_angle=float(self.z_augment_angle),
                              axes=(0, 0, 1)) # z 축의 카메라 
            input_2d_norm = torch.matmul(input_2d_norm,R_rand[:,0:2,0:2])
            # 2차원 카메라 view 를 통해 rotation진행
        preds['keypoints_2d'] = input_2d_norm # 전체 키포인트 값
        preds['kp_mean'] = root # joint 의 중앙 지점
        input_flatten = input_2d_norm.view(-1,self.num_joints_in*2) #2차원 텐서 [Batch , joint X 2]
        feats = self.fe_net(input_flatten[:,:, None, None]) # Batch, n_fully_dim, 1,1
        
         # fully connected layer 통과시켜 feature 뽑음
        # 단일 값을 convolution 진행하기 위해 2차원 tensor로 취급하기 위해 None: None 사용
        # output  = 1
        shape_coeff = self.alpha_layer(feats)[:, :, 0, 0] # 1x1 convolution 진행후 나온 scalar 값
        # pose에 대한 coeff 를 뽑는다 : pose dictionary 의 원소 숫자와 동일하게
        # [Batch X pose dictionary] , 각 Joint 의 형상이 Pose Dictionary 와 비슷한지
        shape_invariant = self.shape_layer(shape_coeff[:, :, None, None])[:, :, 0, 0] # 또다시 2차원 텐서 취급하여 집어넣음
        # shape_layer 의 inde
        # coeff 와 3d pose 를 맞춰주기 위해  3 * num_joint 의 output 
        # 1X1 Convolution 진행
        shape_invariant = shape_invariant.view(ba, self.num_joints_out, 3) # expected Y hat
        # shape 에 대한 정보
        R_log = self.rot_layer(feats)[:,:,0,0] # fully connected layer
        # cam 에 대한 matrix 도 학습을 진행한다. , [Batch X 3]
        R = so3_exponential_map(R_log) # 3차원 Matrix : camera matrix
        # Batch X 3 X 3
        T = R_log.new_zeros(ba, 3)       # no global depth offset

        scale = R_log.new_ones(ba) # batch 만큼의 텐서 생성 , [Batch]
        shape_camera_coord = self.rotate_and_translate(shape_invariant, R, T, scale) # joint matrix 화 R 을 곱함
        # shape_invariant  : Batch X joint X 3 (Pose Dictionary 의 Coeff 에 1X1 conolution 을 진행한 값)
        # R : Batch X 3 X 3 (Camera Matrix)
        # T : Batch X 3 (zero tensor)
        # world 의 점을 2차원 카메라 공간에 투영    
        # cam 에 대한 정보
        # Batch  X  joint X 3
        shape_image_coord = shape_camera_coord[:,:,0:2]/torch.clamp(5 + shape_camera_coord[:,:,2:3],min=1) # Perspective projection
        # z 축 으로의 사영 결과값
        preds['camera'] = R
        preds['shape_camera_coord'] = shape_camera_coord
        preds['shape_coeff'] = shape_coeff # C 
        preds['shape_invariant'] = shape_invariant
        preds['l_reprojection'] = mpjpe(shape_image_coord,input_2d_norm) # joint 에러  
        # pred 로 나온 것과 기존 2d norm 간의 l2 norm
        preds['align'] = align_to_root # boolean       
        
        if self.cycle_consistent: # 복원 loss 의 경우 대체 로스 사용
            preds['l_cycle_consistent'] = self.cycle_consistent_loss(preds)
        return preds

    def cycle_consistent_loss(self, preds, class_mask=None):
        # 재사영 후 
        shape_invariant = preds['shape_invariant'] # hat y
        if preds['align']:    
            shape_invariant_root = shape_invariant - shape_invariant[:,0:1,:]
            # root => head
        else:  
            shape_invariant_root = shape_invariant
        dtype = shape_invariant.type()
        ba = shape_invariant.shape[0]

        n_sample = 4 # 하이퍼파라미터
        # rotate the canonical point
        # generate random rotation around all axes
        R_rand = rand_rot(ba * n_sample,
                dtype=dtype,
                max_rot_angle=3.1415926, # pi 만큼 랜덤하게 돌린다.
                axes=(1, 1, 1))

        unrotated = shape_invariant_root.view(-1,self.num_joints_out,3).repeat(n_sample, 1, 1)
        # 4 배로 늘림
        rotated = torch.bmm(unrotated,R_rand) # 4*B , joint , 3
        rotated_2d = rotated[:,:,0:2] / torch.clamp(5 + rotated[:,:,2:3],min=1) # projection : X'

        repred_result = self.reconstruct(rotated_2d) # Y'

        a, b = repred_result['shape_invariant'], unrotated

        l_cycle_consistent = mpjpe(a,b)

        return l_cycle_consistent

    def reconstruct(self, rotated_2d):
        preds = {}

        # batch size
        ba = rotated_2d.shape[0]
        # reshape and pass to the network ...
        l1_input = rotated_2d.contiguous().view(ba, 2 * self.num_joints_in)

        # pass to network
        feats = self.fe_net(l1_input[:, :, None, None])
        shape_coeff = self.alpha_layer(feats)[:, :, 0, 0]
        shape_pred = self.shape_layer(
                shape_coeff[:, :, None, None])[:, :, 0, 0]

        shape_pred = shape_pred.view(ba,self.num_joints_out,3)
        preds['shape_coeff'] = shape_coeff
        preds['shape_invariant'] = shape_pred

        return preds

    def rotate_and_translate(self, S, R, T, s):
        out = torch.bmm(S, R) + T[:,None,:] # shape X cam matrix + T
        return out

    def normalize_keypoints(self,
                            kp_loc,
                            rescale=1.):
        # center around the root joint
        kp_mean = kp_loc[:, 0, :]
        kp_loc_norm = kp_loc - kp_mean[:, None, :] # [batch , mean , [x,y]]
        kp_loc_norm = kp_loc_norm * rescale
        
        return kp_loc_norm, kp_mean # 각 joint 의 normalize 값과 joint 의 평균값

    def normalize_3d(self,kp):
        ls = torch.norm(kp[:,1:,:],dim=2)
        scale = torch.mean(ls,dim=1)
        kp = kp / scale[:,None,None] * 0.5
        return kp

def pytorch_ge12():
    v = torch.__version__
    v = float('.'.join(v.split('.')[0:2]))
    return v >= 1.2

def conv1x1(in_planes, out_planes, std=0.01):
    """1x1 convolution"""
    cnv = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)

    cnv.weight.data.normal_(0., std)
    if cnv.bias is not None:
        cnv.bias.data.fill_(0.)

    return cnv

class ConvBNLayer(nn.Module):
    def __init__(self, inplanes, planes, use_bn=True, stride=1, ):
        super(ConvBNLayer, self).__init__()

        # do a reasonable init
        self.conv1 = conv1x1(inplanes, planes)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            if pytorch_ge12():
                self.bn1.weight.data.uniform_(0., 1.)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        return out

class ResLayer(nn.Module):
    def __init__(self, inplanes, planes, expansion=4):
        super(ResLayer, self).__init__()
        self.expansion = expansion

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if pytorch_ge12():
            self.bn1.weight.data.uniform_(0., 1.)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if pytorch_ge12():
            self.bn2.weight.data.uniform_(0., 1.)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if pytorch_ge12():
            self.bn3.weight.data.uniform_(0., 1.)
        self.relu = nn.ReLU(inplace=True)
        self.skip = inplanes == (planes*self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip:
            out += residual
        out = self.relu(out)

        return out
