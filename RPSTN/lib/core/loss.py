from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import dsntnn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        # The shape of output: [batch, channel, H, W]
        # print(output.shape, target.shape, target_weight.shape)
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)    # torch.split(tensor, nums, dim=0，1): 将tensor按行(0)或列(1)分割为nums个小tensor 
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class MSESequenceLoss(nn.Module):
    def __init__(self):
        super(MSESequenceLoss, self).__init__()

    def forward(self, inputs, targets):
        T = inputs.shape[1]
        if targets.shape[1] != T:
            f_0 = torch.unsqueeze(targets[:, 0, :, :, :], 1)
            targets = torch.cat([f_0, targets], dim=1)
        return torch.mean(inputs.sub(targets) ** 2)

class CoordinateLoss(nn.Module):
    def __init__(self):
        super(CoordinateLoss, self).__init__()

    def forward(self, heatmaps, coords, targets):
        out = torch.Tensor([31, 31]).cuda()
        batch_size = coords.shape[0]
        n_stages = coords.shape[1]

        if len(targets.shape) != len(coords.shape):
            targets = torch.unsqueeze(targets, 1)
            targets = targets.repeat(1, n_stages, 1, 1)

        targets = (targets.div(255) * 2 + 1) / out - 1

        losses = []
        for i in range(batch_size):
            euc_loss = dsntnn.euclidean_losses(coords[i, :, :, :], targets[i, :, :, :])
            reg_loss = dsntnn.js_reg_losses(heatmaps[i, :, :, :, :], targets[i, :, :, :], sigma_t=1.0)
            losses.append(dsntnn.average_loss(euc_loss + reg_loss))
        return sum(losses) / batch_size


# if __name__ == '__main__':
#     import torch

#     print('Begin Loss.....')
#     use_target_weight = True
#     test_gt = torch.Tensor(torch.randn(10, 5, 14, 64, 64))
#     test_out = torch.Tensor(torch.randn(10, 5, 14, 64, 64))
#     target_weight = torch.randint(0, 2, (10, 5, 14, 1)).float()
#     JMSE = JointsMSELoss(use_target_weight)
#     loss = 0
#     for i in range(5):
#         loss += JMSE(test_out[:, i, :, :, :], test_gt[:, i, :, :, :], target_weight[:, i, :])
#     print(loss / 5)

#     test_gt_2 = test_gt.view(10, -1, 64, 64)
#     test_out_2 = test_out.view(10, -1, 64, 64)
#     target_weight_2 = target_weight.view(10, -1, 1)
#     l = JMSE(test_out_2, test_gt_2, target_weight_2)
#     print(l)
