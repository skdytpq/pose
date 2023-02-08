import torch
from torch.nn import functional as F
import numpy as np


def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim,):
    assert isinstance(heatmaps, torch.Tensor)
    ba = heatmaps.shape[0]
    heatmaps = heatmaps.permute(0,3,1,2) # b,h,w,k -> b k x y
    v_x , v_y = softmax_heat(heatmaps,num_joints , ba) # ba , k , (w,h) , 1
    p_x = torch.arange(1,x_dim + 1)
    p_y = torch.arange(1,y_dim + 1)
    p_x = p_x.repeat(ba,num_joints,1).reshape(ba,num_joints,-1,1) # ba , k , (w,h) , 1 
    p_y = p_y.repeat(ba,num_joints,1).reshape(ba,num_joints,-1,1) # ba , k , (w,h) , 1
    j_x = torch.sum(p_x * v_x,axis=2) # ba , k , 1
    j_y = torch.sum(p_y * v_y,axis=2) # ba , k , 1
    joints = torch.cat([j_x,j_y],axis=2)
    return joints # ba , num_joints , 2, 1


def softmax_heat(heatmaps,num_joints , ba):
    soft_h = torch.sum(torch.exp(heatmaps[:,:,:,:]),(2,3)).reshape(ba,num_joints,1,1)  # b ,k,1,1
    h_k = heatmaps/soft_h
    v_x = h_k.sum(axis=3).reshape(ba,num_joints,-1,1) # b,k,h
    v_y = h_k.sum(axis=2).reshape(ba,num_joints,-1,1) # b,k,w
    return v_x , v_y # b,k,(w,h),1
