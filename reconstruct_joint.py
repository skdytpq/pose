from __future__ import absolute_import
import numpy as np

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from functools import reduce
from ITES.common.function import so3_exponential_map
from ITES.common.camera import *
from ITES.common.function import *
from ITES.common.loss import *





class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, grouped_order, restored_order, group_size, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.non_local = _GraphNonLocal(output_dim, grouped_order, restored_order, group_size)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x1 = self.gconv(x).transpose(1, 2)
        x2 = self.non_local(x).transpose(1,2)
        x = x1 + x2
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        x = self.relu(x)
        return x

class _Simple_GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, grouped_order, restored_order, group_size, p_dropout=None):
        super(_Simple_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2) 
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        x = self.relu(x)
        return x

class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, grouped_order, restored_order, group_size,p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, grouped_order, restored_order, group_size, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, grouped_order, restored_order, group_size, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out

class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out

class Student_net(nn.Module): # 2D input 과의 차이 계산 하기 위해 사용 N , 17 , 3  ㅑ
    def __init__(self, adj, hid_dim, coords_dim=(2, 1), num_layers=4, nodes_group=None, p_dropout=None):
        super(Student_net, self).__init__()
        group_size = len(nodes_group[0])
        assert group_size > 1

        grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
        restored_order = [0] * len(grouped_order)
        for i in range(len(restored_order)):
            for j in range(len(grouped_order)):
                if grouped_order[j] == i:
                    restored_order[i] = j
                    break
        _gconv_input = [_Simple_GraphConv(adj, coords_dim[0], hid_dim, grouped_order, restored_order, group_size, p_dropout=p_dropout)]
        # output = 17X2 Joint matrix
        _gconv_layers = []
        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, grouped_order, restored_order, group_size,p_dropout))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[0], adj)

    def predict(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out

    def forward(self,input_2d):
        preds = {}
        ba = input_2d.shape[0]
        dtype = input_2d.type()
        preds['keypoints_2d'] = input_2d
        input_flatten = input_2d
        depth = self.predict(input_flatten)
        preds['reconstruct'] = depth # 3차원 공간이 나옴
        return preds
    
    def reconstruct(self,input_2d):
        input_flatten = input_2d
        depth = self.predict(input_flatten)
        shape_3d = torch.cat((input_2d * torch.clamp(depth+5,min=1),depth),dim=2)
        return shape_3d

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """
    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float)) # 2 X 2 X 128
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj # joint 노드에 대한 adj matrix
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float)) # zero가 아닌 update parameter
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0]) # 17 X 2 , 2 X 2
        h1 = torch.matmul(input, self.W[1]) # 17 X 2 , 2 X 2

        adj = -9e15 * torch.ones_like(self.adj).to(input.device) # numjoint  X numjoint
        adj[self.m] = self.e # 0보다 큰 adj index를 1로 채움 
        adj = F.softmax(adj, dim=1) # 확률 값으로 변환

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device) # 17 size 의 대각행렬
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1) # 각 노드 자신에 대한 값을 한번 곱한 후 나머지 연결 값과도 곱함
        # 17 X 17 * 17 X 17 , 17 X 2  -> 17 X 2 + 17 X 17 * (17 X 17) * 17 X 2  -> 17 X 2 
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class _NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=1, bn_layer=True):
        super(_NonLocalBlock, self).__init__() 
        # input = 17X2
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        assert self.inter_channels > 0

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        elif dimension == 1:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d
        else:
            raise Exception('Error feature dimension.')

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0) # Convolution layer 를 거친다.
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal_(self.W[0].weight)
            nn.init.constant_(self.W[0].bias, 0)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample > 1:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=sub_sample))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=sub_sample))

    def forward(self, x):
        batch_size = x.size(0)  # x: (b, c, t, h, w)  ?

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1) N  : joint 
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2) # N
        w = phi_x.size(3) # N
        theta_x = theta_x.expand(-1, -1, -1, w) # 마지막 차원의 size 를 joint_size 로 맞춤 -> joint 만큼의 값 복사 
        phi_x = phi_x.expand(-1, -1, h, -1) # 3번째 차원 사이즈를 joint_size 로 맞춤

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w) # b , 17 , 17

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x) # b X 17 X 17 , b X c X 17
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z # batch X 17 X 2

class GraphNonLocal(_NonLocalBlock):
    def __init__(self, in_channels, inter_channels=None, sub_sample=1, bn_layer=True):
        super(GraphNonLocal, self).__init__(in_channels, inter_channels=inter_channels, dimension=1,
                                            sub_sample=sub_sample, bn_layer=bn_layer)
