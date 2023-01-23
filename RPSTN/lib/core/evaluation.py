
import torch
import numpy as np
from prettytable import PrettyTable


def accuracy(inputs, targets, r=0.2):
    # print(inputs.shape)
    batch_size = inputs.shape[0]
    n_stages = inputs.shape[1]
    n_joints = inputs.shape[2]

    inputs = inputs.detach()
    targets = targets.detach()

    if targets.shape[1] != n_stages:
        f_0 = torch.unsqueeze(targets[:, 0, :, :, :], 1)
        targets = torch.cat([f_0, targets], dim=1)

    n_correct = 0
    n_total = batch_size * n_stages * n_joints * n_stages

    for i in range(batch_size):
        gt = get_preds(targets[i, :, :, :, :])
        preds = get_preds(inputs[i, :, :, :, :])
        # print(gt.shape, preds.shape)

        for j in range(n_stages):
            w = gt[j, :, 0].max() - gt[j, :, 0].min()
            h = gt[j, :, 1].max() - gt[j, :, 1].min()
            threshold = r * max(w, h)

            scores = torch.norm(preds.sub(gt), dim=2).view(-1)
            n_correct += scores.le(threshold).sum()
    #n_correct = n_correct / n_stages 
    return float(n_correct) / float(n_total)


def coord_accuracy(inputs, gt, r=0.2):
    batch_size = inputs.shape[0]
    n_stages = inputs.shape[1]
    n_joints = inputs.shape[2]

    inputs = inputs.detach()
    gt = gt.detach()

    n_correct = 0
    n_total = batch_size * n_stages * n_joints

    for i in range(batch_size):
        w = gt[i, :, 0].max() - gt[i, :, 0].min()
        h = gt[i, :, 1].max() - gt[i, :, 1].min()
        threshold = r * max(w, h)

        curr_gt = torch.unsqueeze(gt[i], 0).repeat(n_stages, 1, 1)
        scores = torch.norm(inputs[i].sub(curr_gt), dim=2).view(-1)
        n_correct += scores.le(threshold).sum()

    return float(n_correct) / float(n_total)


# Source: https://github.com/bearpaw/pytorch-pose/blob/master/pose/utils/evaluation.py
def get_preds(scores):
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def get_final_preds(batch_heatmaps):
    coords = get_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_weight = batch_heatmaps.shape[3]

    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]   # 第n个batch/图像中的第p个关节点
            px = int(torch.floor(coords[n][p][0] + 0.5))
            py = int(torch.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_weight-1 and 1 < py < heatmap_height - 1:
                dif = torch.Tensor([hm[py][px+1] - hm[py][px-1], hm[py+1][px] - hm[py-1][px]]).cuda()
                coords[n][p] += torch.sign(dif) * .25
    preds = coords.clone()  * 4
    return preds



def calc_PCK(preds, targets, bboxes, joint_vis, seqTrain, normTorso=None):
    '''
        0: lankle       1: lknee
        2: lhip         3: rhip
        4: rknee        5: rankle
        6: relbow       7: rwrist
        8: neck         9: head
        10: lwrist      11: lelbow
        12: lshoulder   13: rshoulder
    '''
    # preds = np.delete(preds, 8, axis=2)
    # targets = np.delete(targets, 8, axis=2)
    # joint_vis = np.delete(joint_vis, 8, axis=2)

    # print(preds)
    # print('=' * 20)
    # print(targets)
    # targets = targets * 4
    joint_vis[:, :, -1] = 0
    gtJointOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # gtJointOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    thresh = 0.2

    sample_num = targets.shape[0]
    
    HitPoint = np.zeros((sample_num, len(gtJointOrder)))
    visible_joint = np.zeros((sample_num, len(gtJointOrder)))
    for sample in range(0, sample_num):
        # print('test sample:', sample)
        test_gt = targets[sample]
        test_out = preds[sample]
        visibility = joint_vis[sample]
        bbox = bboxes[sample]
    
        nfr = seqTrain

        # num_frame = test_gt.shape[0]
        seqError = torch.zeros(nfr, len(gtJointOrder))
        seqThresh = torch.zeros(nfr, len(gtJointOrder))

        for frame in range(0, nfr):
            gt = test_gt[frame] # 13x2
            pred = test_out[frame] # 13x2
            # vis = visibility[frame] # 1x13
            if normTorso:
                bodysize = torch.norm(gt[7] - gt[2])
                if bodysize < 1:
                    bodysize = torch.norm(pred[7] - pred[2])
            else:
                bodysize = torch.max(bbox[frame,2]-bbox[frame, 0], bbox[frame, 3] - bbox[frame, 1])

            error_dis = torch.norm(gt-pred, p=2, dim=1, keepdim=False)
            bodysize = bodysize.float()
            seqError[frame] = error_dis
            seqThresh[frame] = (bodysize*thresh) * torch.ones(len(gtJointOrder))

        vis = visibility[0:nfr, :]
        visible_joint[sample] = np.sum(vis.numpy(), axis=0)
        less_than_thresh = np.multiply(seqError.numpy()<=seqThresh.numpy(), vis.numpy())
        # visibleJoint = np.sum(visibility.numpy(), axis=0)
        HitPoint[sample] = np.sum(less_than_thresh, axis=0)

    # finalPCK = np.divide(np.sum(HitPoint, axis=0), np.sum(np.sum(Visbility.numpy(), axis=1), axis=0))
    print(np.sum(HitPoint[:-1], axis=0))
    print(np.sum(visible_joint, axis=0))
    finalPCK = np.divide(np.sum(HitPoint, axis=0)[:-1], np.sum(visible_joint, axis=0)[:-1])
    finalMean = np.mean(finalPCK)

    table = PrettyTable(['Joints', 'Score'])
    table.add_row(['Head', finalPCK[0]])
    table.add_row(['Shoulder', 0.5*(finalPCK[1]+finalPCK[2])])
    table.add_row(['Elbow', 0.5*(finalPCK[3]+finalPCK[4])])
    table.add_row(['Wrist', 0.5*(finalPCK[5]+finalPCK[6])])
    table.add_row(['Hip', 0.5*(finalPCK[7]+finalPCK[8])])
    table.add_row(['Knee', 0.5*(finalPCK[9]+finalPCK[10])])
    table.add_row(['Ankle', 0.5*(finalPCK[11]+finalPCK[12])])
    table.add_row(['Mean', finalMean])
    print(table)
    return finalPCK, finalMean

# if __name__ == '__main__':
#     preds = torch.randn(16, 5, 14, 2)
#     gt = torch.randn(16, 5, 14, 2)
#     bbox = torch.randn(16, 5, 4)
#     vis = torch.randint(0,2, (16, 5, 14))
#     print(vis[0, 4, :])

#     vis_0 = vis.clone()
#     vis_0[:, :, -1] = 0
#     print(vis_0[0, 4, :])
#     calc_PCK(preds, gt, bbox, vis, 5)