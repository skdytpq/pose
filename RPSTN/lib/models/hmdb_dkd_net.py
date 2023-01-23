import numpy as np
import torch
import torch.nn as nn
import sys
import pdb
from models.module.ad_conv import *
from models.hmdb_pose_resnet import *
from models.module.jre_module import *


class DKD_Network(nn.Module):
    def __init__(self, cfg, is_visual, inplanes=64, is_train=True, T=5):
        super(DKD_Network, self).__init__()
        self.is_train = is_train
        self.pose_net = get_pose_net(cfg, 'pose_init', is_train=True)
        self.fea_ext = get_pose_net(cfg, 'fea_ext', is_train=True)

        self.T = T
        self.inplanes = inplanes
        self.heatmap_w, self.heatmap_h = cfg.DATASET.HEATMAP_SIZE
        self.fea_dim = 256
        self.joint_num = cfg.JHMDB.NUM_JOINTS
        self.initial_weights = None
        self.is_visual = is_visual

        self.conv1x1_reduce_dim = nn.Conv2d(self.fea_dim + self.joint_num, self.fea_dim, 1, 1)
        self.pose_conv1x1_V = nn.Conv2d(self.fea_dim, self.fea_dim, 1, 1)
        self.pose_conv1x1_U = nn.Conv2d(self.fea_dim, self.joint_num, 1, 1)

        self.bn_u = nn.BatchNorm2d(self.joint_num)

        self.bn = nn.BatchNorm2d(self.fea_dim)
        self.relu = nn.ReLU(inplace=True)

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
        self.non_local_conv2d = JREModule(self.joint_num, is_visual, sub_sample=False, in_split=False, is_joint=True, use_weight=False)

    def forward(self, x, gt_x):
        x = torch.split(x, 1, dim=1)
        x_0 = torch.squeeze(x[0], 1)

        heatmap_0 = self.pose_net(x_0)
        featmap_0 = self.fea_ext(x_0)

        heatmap_0 = self.non_local_conv2d(heatmap_0)
        heatmap_0 = self.bn_u(heatmap_0)

        beliefs = [heatmap_0]
        gt_relation, pred_relation = [], []

        h_prev, f_prev = heatmap_0, featmap_0
        for t in range(1, self.T):
            x_t = torch.squeeze(x[t], 1)

            f = self.fea_ext(x_t)
            f = self.pose_conv1x1_V(f)
            f = self.bn(f)

            con_fea_maps = torch.cat([h_prev, f_prev], dim=1)

            con_fea_maps = self.conv1x1_reduce_dim(con_fea_maps)
            con_fea_maps = self.bn(con_fea_maps)

            dkd_features = self.dkd_param_adapter(con_fea_maps)
            f_b, f_c, f_h, f_w = f.shape

            pose_adaptive_conv = AdaptiveConv2d(f_b * f_c,
                                            f_b * f_c,
                                            7, padding=3,
                                            groups=f_b * f_c,
                                            bias=False)
            con_map = pose_adaptive_conv(f, dkd_features)
            con_map = self.bn(con_map)

            confidences = self.pose_conv1x1_U(con_map)
            h = self.bn_u(confidences)
            
            h = self.non_local_conv2d(h)
            h = self.bn_u(h)

            beliefs.append(h)
            h_prev, f_prev = h, f
        confidence_maps = torch.stack(beliefs, 1)
        return confidence_maps

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
