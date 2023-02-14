import torch
from torch.nn import functional as F
import numpy as np
import pdb

def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim,):
    assert isinstance(heatmaps, torch.Tensor) # b,Seq,h,w,k 
    #heatmaps = heatmaps.view(-1,num_joints,heatmaps.shape[-2],heatmaps.shape[-1])
    ba = heatmaps.shape[0]
    seq = heatmaps.shape[1]
    joints = torch.zeros([ba,seq,num_joints,2]).to(device)
    for i in range(seq): # seq 끼리 계산하여 tensor 차원 맞추기
        heatmaps = heatmaps[:,i,:,:,:].rehsape(ba,num_joints,heatmaps.shape[-2],heatmaps.shape[-1])
    #heatmaps = heatmaps.permute(0,3,1,2) # b,h,w,k -> b k x y # 8 , 5
        device = torch.device("cuda:0")
        v_x , v_y = softmax_heat(heatmaps,num_joints , ba) # ba , k , (w,h) , 1\
        output = soft_ar(heatmaps)
        v_x , v_y = v_x.to(device),v_y.to(device)
        p_x = torch.arange(1,x_dim + 1).to(device)
        p_y = torch.arange(1,y_dim + 1).to(device)
        p_x = p_x.repeat(ba,num_joints,1).reshape(ba,num_joints,-1,1) # ba , k , (w,h) , 1 
        p_y = p_y.repeat(ba,num_joints,1).reshape(ba,num_joints,-1,1) # ba , k , (w,h) , 1

        j_x = torch.sum(p_x * v_x,axis=2) # ba , k , 1
        j_y = torch.sum(p_y * v_y,axis=2) # ba , k , 1
        joints_ = torch.cat([j_x,j_y],axis=2)
        joints[:,i,:,0] = j_x[:,:,:]
        joints[:,i,:,1] = j_y[:,:,:]
    pdb.set_trace()
    joints = output
    return joints_ # ba , num_joints , 2, 1


def softmax_heat(heatmaps,num_joints , ba):
    heatmaps = heatmaps.mul(100)
    soft_h = torch.sum(torch.exp(heatmaps[:,:,:,:]),(2,3)).reshape(ba,num_joints,1,1)  # b ,k,1,1
    h_k = torch.exp(heatmaps[:,:,:,:])/soft_h
    v_x = h_k.sum(axis=3).reshape(ba,num_joints,-1,1) # b,k,h
    v_y = h_k.sum(axis=2).reshape(ba,num_joints,-1,1) # b,k,w
    return v_x , v_y # b,k,(w,h),1

def soft_ar(heatmap):
    heatmap = heatmap.mul(100)
    batch_size, num_channel, height, width = heatmap.size()
    device: str = heatmap.device

    softmax: torch.Tensor = F.softmax(
        heatmap.view(batch_size, num_channel, height * width), dim=2
    ).view(batch_size, num_channel, height, width)

    xx, yy = torch.meshgrid(list(map(torch.arange, [width, height])))
    
    approx_x = (
        softmax.mul(xx.float().to(device))
        .view(batch_size, num_channel, height * width)
        .sum(2)
        .unsqueeze(2)
    )
    approx_y = (
        softmax.mul(yy.float().to(device))
        .view(batch_size, num_channel, height * width)
        .sum(2)
        .unsqueeze(2)
    )
    pdb.set_trace()
    output = [approx_x, approx_y] #if self.return_xy else [approx_y, approx_x]
    output = torch.cat(output, 2)
    return output