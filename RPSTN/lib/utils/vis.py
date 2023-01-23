from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import torch
import matplotlib.pyplot as plt

from .evaluate import get_max_preds
from matplotlib.backends.backend_pdf import PdfPages


def save_batch_image_with_joints(dataset, batch_image, batch_joints, gt_val, batch_joints_vis,
                                 file_name, nrow=5, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # joint_colors = []
    if dataset == 'PennAction':
        skeleton = [[1, 2], [2, 4], [4, 6], [1, 3], [3, 5], [2, 8], [7, 8],
                    [8, 10], [10, 12], [1, 7], [7, 9], [9, 11]]

    # colors = ['#663399', '#99CCFF', '#99CCFF', '#CCFFFF', '#CCFF99', '#CCFFFF', '#CCFF99',
    #         '#FFFF66', '#FFCC99 ', '#FF9900', '#FFFF66', '#FFCC99 ', '#FF9900']

        joint_colors = [[51, 0, 255], [102, 102, 255], [102, 102, 255], [153, 51, 0], [153, 51, 0], [0, 102, 51], [0, 102, 51],
                [102, 255, 255], [102, 255, 255], [51, 153, 255], [51, 153, 255], [153, 204, 255], [153, 204, 255]]

        colors = [[0, 102, 51], [0, 102, 204], [153, 51, 0], [0, 102, 204], [153, 51, 0], [0, 102, 51], [0, 102, 51],
                [102, 255, 255], [153, 204, 255], [0, 102, 51], [102, 255, 255], [153, 204, 255]]
    elif dataset == 'Sub-JHMDB':
        skeleton = [[0, 2], [0, 3], [0, 4], [3, 7], [3, 11], [4, 8], [8, 12], [0, 1], [1, 5],
                    [5, 9], [9, 13], [1, 6], [6, 10], [10, 14]]
        joint_colors = [[51, 0, 255], [102, 102, 255], [102, 102, 255], [153, 51, 0], [153, 51, 0], [0, 102, 51], [0, 102, 51],
                [102, 255, 255], [102, 255, 255], [51, 153, 255], [51, 153, 255], [153, 204, 255], [153, 204, 255] , [153, 204, 255], [153, 204, 255]]

        colors = [[0, 102, 51], [0, 102, 204], [153, 51, 0], [0, 102, 204], [153, 51, 0], [0, 102, 51], [0, 102, 51],
                [102, 255, 255], [153, 204, 255], [0, 102, 51], [102, 255, 255], [153, 204, 255], [153, 204, 255], [153, 204, 255]]

    images = torch.zeros(batch_image.size(0), 3, 256, 256)

    GPimages = torch.zeros(2*batch_image.size(0), 3, 256, 256)

    for i in range(batch_image.size(0)):
        image = batch_image[i].clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        resized_image = np.array(cv2.resize(image,(256, 256)), dtype='uint8')
        # print(resized_image)
        images[i] = torch.from_numpy(resized_image.transpose(2, 0, 1))

    ij = 0
    gp = 0
    while ij < images.size(0):
        GPimages[gp] = images[ij]
        GPimages[gp+1] = images[ij]
        ij += 1
        gp += 2

    grid = torchvision.utils.make_grid(GPimages, nrow, padding, True)
    # print(grid[0])
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = np.array(ndarr.copy(), dtype='uint8')
    # print(ndarr[0])

    nmaps = images.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(images.size(2) + padding)
    width = int(images.size(3) + padding)

    # print(height, width)
    gt_row = [0, 2, 4, 6, 8, 10, 12, 14]
    pred_row = [1, 3, 5, 7, 9, 11, 13, 15]

    k = 0
    for y in gt_row:
        for x in range(xmaps):
            if k >= batch_joints_vis.shape[0]:
                break

            joints = gt_val[k]  
            # joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            col = 0
            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]>0:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, joint_colors[col], 2)
                col = col + 1
            # col = 0
            # for sk in skeleton:
            #     pos1 = (int(joints[sk[0], 0]), int(joints[sk[0], 1]))
            #     pos2 = (int(joints[sk[1], 0]), int(joints[sk[1], 1]))
            #     if pos1[0]>0 and pos1[1] >0 and pos2[0] >0 and pos2[1] > 0 and joints_vis[sk[0]] > 0 and joints_vis[sk[1]] > 0:
            #         cv2.line(ndarr, pos1, pos2, colors[col], 2)
            #     col = col + 1
            k = k + 1

    k=0
    for y in pred_row:
        for x in range(xmaps):
            if k >= batch_joints_vis.shape[0]:
                break

            # joints = gt_val[k]  
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            col = 0
            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]>0:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, joint_colors[col], 2)
                col = col + 1
            # col = 0
            # for sk in skeleton:
            #     pos1 = (int(joints[sk[0], 0]), int(joints[sk[0], 1]))
            #     pos2 = (int(joints[sk[1], 0]), int(joints[sk[1], 1]))
            #     if pos1[0]>0 and pos1[1] >0 and pos2[0] >0 and pos2[1] > 0 and joints_vis[sk[0]] > 0 and joints_vis[sk[1]] > 0:
            #         cv2.line(ndarr, pos1, pos2, colors[col], 2)
            #     col = col + 1
            k = k + 1
    # pp = PdfPages(file_name)
    # plt.imshow(ndarr[:, :, ::-1])
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    # pp.savefig()
    # pp.close()
    cv2.imwrite(file_name, ndarr.astype(np.uint8))


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
