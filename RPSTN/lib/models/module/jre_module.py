import torch
from torch import nn
import math
import numpy as np
import pdb
from torch.nn import functional as F
from torch.nn.parameter import Parameter

# Manual graph
strucutral_matrix = np.array([
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                    ])

class _JREModule(nn.Module):
    def __init__(self, in_channels, is_visual, inter_channels=None, dimension=3, sub_sample=True, \
                    bn_layer=True, in_split=True, is_joint=True, use_weight=False):
        '''
        # in_channels: input dimention
        # is_visual: do visualization (yes or no)
        # dimension: the size of image
        # sub_sample: do the sub_sample
        # bn_layer: if use teh batch normalization layer
        # in_split: split features to N groups
        # is_joint: joint-level attention (True) / pixel-level attention (False)
        # use_weight: if add the manual graph
        '''

        super(_JREModule, self).__init__()

        assert dimension in [1, 2, 3]

        self.is_visual = is_visual
        self.joint_num = 13
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.is_joint = is_joint
        self.use_weight = use_weight

        if self.use_weight:
            self.att = Parameter(torch.FloatTensor(strucutral_matrix))
            
        if self.inter_channels is None:
            if in_split:
                self.inter_channels = in_channels // 2
            else:
                self.inter_channels = in_channels
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d


        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)



    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.inter_channels)
        self.att.data.uniform_(-stdv, stdv)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size, c, h, w = x.size()
        np.save('result/features/conv1.npy', self.g(x).detach().cpu().numpy())
        pdb.set_trace()
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   # [b, c, h*w]

        if self.is_joint:
            theta_x = x.view(batch_size, self.inter_channels, -1)   # [b, c, h*w]
            phi_x = x.view(batch_size, self.inter_channels, -1)
            phi_x = phi_x.permute(0, 2, 1).contiguous()
        else:
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)  # [b, h*w, c]
            theta_x = theta_x.permute(0, 2, 1)  # [b, h*w, c]

        f = torch.matmul(theta_x, phi_x)
        if self.use_weight:
            f = torch.matmul(self.att, f)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = y
        W_y = self.W(y)
        z = W_y + x

        return z


class JREModule(_JREModule):
    def __init__(self, in_channels, is_visual, inter_channels=None, sub_sample=True, bn_layer=True, in_split=True, is_joint=False, use_weight=False):
        super(JREModule, self).__init__(in_channels, is_visual,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, in_split=in_split, is_joint=is_joint, use_weight=use_weight)
