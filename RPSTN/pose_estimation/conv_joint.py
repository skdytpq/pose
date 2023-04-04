import torch
from torch.nn import functional as F
import numpy as np
import pdb
import torch.nn as nn

def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim,):
    assert isinstance(heatmaps, torch.Tensor) # b,Seq,h,w,k 
    #heatmaps = heatmaps.view(-1,num_joints,heatmaps.shape[-2],heatmaps.shape[-1])
    device = torch.device("cuda:0")
    ba = heatmaps.shape[0]
    seq = heatmaps.shape[1]
    heatmaps_ = heatmaps # b 13 64 64
    joints = torch.zeros([ba,seq,num_joints,2]).to(device)
    for i in range(seq): # seq 끼리 계산하여 tensor 차원 맞추기
        heatmaps_ = heatmaps[:,i,:,:,:].reshape(ba,num_joints,heatmaps.shape[-2],heatmaps.shape[-1])
        output = soft_ar(heatmaps_)
        j_x = output[:,:,1]
        j_y = output[:,:,0]
        joints[:,i,:,0] = j_x[:,:].reshape(ba,num_joints)
        joints[:,i,:,1] = j_y[:,:].reshape(ba,num_joints)
    joints = joints.reshape(-1,num_joints,2)
    heat = heatmaps_.reshape(-1,num_joints,heatmaps.shape[-2],heatmaps.shape[-1])
    heat = heat.cuda()
    heatconv(heat)
    pdb.set_trace()
    return joints


class heatconv(nn.Module):
    def __init__(self, heatmap_size = 64, n_fully_connected=1024, n_layers=4 ,num_joints = 13):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.n_fully_connected = 1024#n_fully_connected
        self.n_layers = 4#n_layers
        self.num_joints =13 # num_joints

        self.fe_net = nn.Sequential(
         ConvBNLayer(self.num_joints,self.n_fully_connected,True),
         ResLayer(self.n_fully_connected , self.n_fully_connected/4)
         ,ResLayer(self.n_fully_connected /4 ,self.n_fully_connected/16)) # Convolution Batchnormailization fully connected layer
        self.avg = nn.AdaptiveAvgPool2d(self.n_fully_connected/16)
                                   
    def forward(self,heatmap):
        ba = heatmap.shape[0]
        pdb.set_trace()
        confidence = self.fe_net(heatmap)
        pdb.set_trace()
        confidence = self.avg(confidence)
        conf = confidence.reshape(ba,-1)
        return conf






class ConvBNLayer(nn.Module):
    def __init__(self, inplanes, planes, use_bn=True, stride=1, ):
        super(ConvBNLayer, self).__init__()

        # do a reasonable init
        self.conv1 = conv3x3(inplanes, planes)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        return out
        


def conv3x3(in_planes, out_planes, std=0.01):
    """1x1 convolution"""
    cnv = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=3,padding =  1)

    cnv.weight.data.normal_(0., std)
    if cnv.bias is not None:
        cnv.bias.data.fill_(0.)

    return cnv

class ResLayer(nn.Module):
    def __init__(self, inplanes, planes, expansion=4):
        super(ResLayer, self).__init__()
        self.expansion = expansion

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


def soft_ar(heatmap):
    heatmap = heatmap.mul(50)
    batch_size, num_channel, height, width = heatmap.size()
    # Batch , channe l , 64 , 64
    device: str = heatmap.device

    softmax: torch.Tensor = F.softmax(
        heatmap.view(batch_size, num_channel, height * width), dim=2
    ).view(batch_size, num_channel, height, width)
    # B , chaneel , 64X64 
    xx, yy = torch.meshgrid(list(map(torch.arange, [width, height])))
    xx  = xx + 1
    yy = yy + 1
    # 64,64 [0~64]
    approx_x = (
        softmax.mul(xx.float().to(device))
        .view(batch_size, num_channel, height * width) # 8,16,1
        .sum(2)
        .unsqueeze(2)
    )
    approx_y = (
        softmax.mul(yy.float().to(device))
        .view(batch_size, num_channel, height * width)
        .sum(2)
        .unsqueeze(2)
    )
    output = [approx_x, approx_y] #if self.return_xy else [approx_y, approx_x]
    output = torch.cat(output, 2)
    return output

    