import numpy as np
import torch
import torch.nn as nn
import sys
import pdb
from models.module.ad_conv import *
from models.pose_resnet import *
from models.module.jre_module import *

# The JRPSP input includes JRE output, adjacent features 
class DKD_Network(nn.Module):
    def __init__(self, cfg, is_visual, inplanes=64, is_train=True, T=5):
        '''
        # cfg: config
        # is_visual: if do the visualization
        # inplanes: input dimention
        # is_train: do train (true) or test (false)
        # T: the number of input frames
        '''
        super(DKD_Network, self).__init__()
        self.is_train = is_train
        self.pose_net = get_pose_net(cfg, 'pose_init', is_train=True) #
        self.fea_ext = get_pose_net(cfg, 'fea_ext', is_train=True) # 13 
        # 동일한 네트워크 구조를 통과하나 parameter 가 다름, Resnet
        self.T = T
        self.inplanes = inplanes
        self.heatmap_w, self.heatmap_h = cfg.DATASET.HEATMAP_SIZE
        self.fea_dim = 256
        self.joint_num = cfg.DATASET.NUM_JOINTS
        self.initial_weights = None
        self.is_visual = is_visual

        # Reduce the dimention before enter the SPSP module
        self.conv1x1_reduce_dim = nn.Conv2d(self.fea_dim + self.joint_num, self.fea_dim, 1, 1)
        self.pose_conv1x1_V = nn.Conv2d(self.fea_dim, self.fea_dim, 1, 1)
        self.pose_conv1x1_U = nn.Conv2d(self.fea_dim, self.joint_num, 1, 1)

        self.bn_u = nn.BatchNorm2d(self.joint_num)
        self.bn = nn.BatchNorm2d(self.fea_dim)
        self.relu = nn.ReLU(inplace=True)

        # SPSP module
        self.dkd_param_adapter = nn.Sequential(
                                    nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, stride=1, padding=0),
                                    nn.BatchNorm2d(self.fea_dim),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                                    nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(self.fea_dim),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                                    nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(self.fea_dim),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                    
                                    nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(self.fea_dim)
                                    )
        # SKL module
        self.non_local_conv2d = JREModule(self.joint_num, is_visual, sub_sample=False, in_split=False, is_joint=True, use_weight=False)
        self.iter = 0

    def forward(self, x):
        x = torch.split(x, 1, dim=1)
        x_0 = torch.squeeze(x[0], 1) # 1st frame

        heatmap_0 = self.pose_net(x_0) # pose estimator via pos_resnet, JRE Module 
        featmap_0 = self.fea_ext(x_0) # final feature dim = number of joint

        heatmap_0 = self.non_local_conv2d(heatmap_0) # R(P(I))
        heatmap_0 = self.bn_u(heatmap_0)
        # make m'
        beliefs = [heatmap_0]
        gt_relation, pred_relation = [], []

        h_prev, f_prev = heatmap_0, featmap_0
        # h_prev : joint information 
        # t = 5
        for t in range(1, self.T):
            self.iter += 1
            x_t = torch.squeeze(x[t], 1)

            f = self.fea_ext(x_t) # posenet -> Resnet 기반 모델을 지나감
            f = self.pose_conv1x1_V(f) # 1X1XC convolution 을 통해 pose information 추출
            # 1X1XC convonlution 은 논문에서 나오지 않음
            # 현재의 시점에서의 feature 생성
            f = self.bn(f) # pose information 

            con_fea_maps = torch.cat([h_prev, f_prev], dim=1)  # 이전 frame 에서 가져온 정보
            con_fea_maps = self.conv1x1_reduce_dim(con_fea_maps) # feature dimension 인 C 로 convolution 진행
            con_fea_maps = self.bn(con_fea_maps) 
            ## JPRSP : concat을 한 feature들의 조합된 형태가 학습을 진행하기 위해 Propagation을 위한 층 생성
            dkd_features = self.dkd_param_adapter(con_fea_maps) #  정보 압축 SPSP 모듈 256X7X7
            # 이전 시점의 Feautre map 과 Heatmap 을 concat 하여(t 시점) X 생성 <- semantic feature
            # feature f 의 시점은 이 전 frame 의 시점이다.
            # F(i_t+1) heatmap joint 
            f_b, f_c, f_h, f_w = f.shape # batch * channel
            pose_adaptive_conv = AdaptiveConv2d(f_b * f_c,
                                            f_b * f_c,
                                            7, padding=3,
                                            groups=f_b * f_c,
                                            bias=False)
            con_map = pose_adaptive_conv(f, dkd_features) # 논문에서의 Global matching 
            confidences = self.pose_conv1x1_U(con_map) # M t+1 generate
            h = self.bn_u(confidences)  # 최종 출력 
            
            final_h = self.non_local_conv2d(h) # JRE 를 거쳐 최종적인 M_t+1 시점의 Heat map 생성
            # confidence 를 JRE에 t+1 시점으로 간주하여 넣음 
            final_h = self.bn_u(final_h)
            
            beliefs.append(final_h)
            h_prev, f_prev = final_h, f
        confidence_maps = torch.stack(beliefs, 1)
        return confidence_maps # B, Frame , Joint num * Joint num

    def init_weights(self):
        for name, m in self.dkd_param_adapter.named_modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, val=0.0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, val=1.0)
                torch.nn.init.constant_(m.bias, val=0.0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.weight, val=0.0)

        if isinstance(self.conv1x1_reduce_dim, nn.Conv2d):
            torch.nn.init.kaiming_normal_(self.conv1x1_reduce_dim.weight, mode='fan_out', nonlinearity='relu')
            if self.conv1x1_reduce_dim.bias is not None:
                torch.nn.init.constant_(self.conv1x1_reduce_dim.bias, val=0.0)
                
        for m in self.pose_conv1x1_U.named_modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, val=0.0)
        for m in self.pose_conv1x1_V.named_modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, val=0.0)
        for m in self.non_local_conv2d.named_modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def get_dkd_net(cfg, is_visual, is_train=True):
    dkd_net = DKD_Network(cfg, is_visual, is_train)
    if is_train:
        dkd_net.init_weights()
    return dkd_net


# if __name__ == '__main__':
#     import sys
#     sys.path.append('../../lib')
#     from thop import profile
#     from core.config import config
#     test_out = torch.Tensor(torch.randn(8, 5, 3, 256, 256))
#     gt_x = torch.Tensor(torch.randn(8, 5, 3, 64, 64))
# #     print(test_out.shape)
# #     print('Loading dkd')
#     dkd = get_dkd_net(config)
#     macs, params = profile(dkd, inputs=(test_out, gt_x))
#     print('FLOPs: ', macs, ' Params: ', params)
#     h = dkd(test_out)
#     print(h.shape)