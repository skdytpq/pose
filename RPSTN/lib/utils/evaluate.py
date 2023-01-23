from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import pdb
from prettytable import PrettyTable


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))

    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets + 1)
            else:
                dists[c, n] = -1

    return dists


def dist_acc(dists, threshold=0.5):
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()

    if num_dist_cal > 0:
        return np.less(dists[dist_cal], threshold).sum() * 1.0 / num_dist_cal
    else:
        return -1


def get_max_preds(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # print('In get_max_preds: ', batch_size, num_joints, width)

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask

    return preds, maxvals


def accuracy(outputs, targets, thr_PCK, thr_PCKh, dataset, bbox, hm_type='gaussian', threshold=0.35, normTorso=True):
    idx = list(range(outputs.shape[1]))  # 14 joints
    # bbox = bbox
    # print(bbox)
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(outputs)
        target, _ = get_max_preds(targets)
        h = outputs.shape[2]
        w = outputs.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    # print(pred.shape, target.shape, bbox.shape)
    # heat_dist = calc_dists(outputs, targets, norm)
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx)))
    avg_acc = 0
    cnt = 0
    visible = np.zeros((len(idx)))

    if dataset == 'Penn_Action':
        for i in range(len(idx)):
            acc[i] = dist_acc(dists[idx[i]])
            # acc[i] = dist_acc(heat_dist[idx[i]])
            if acc[i] >= 0:
                avg_acc = avg_acc + acc[i]
                cnt += 1
                visible[i] = 1
            else:
                acc[i] = 0

        avg_acc = avg_acc / cnt if cnt != 0 else 0

    elif dataset == 'Sub-JHMDB':
        for i in range(len(idx)):
            # if i != 0 or i != 1:
            acc[i] = dist_acc(dists[idx[i]])
            # acc[i] = dist_acc(heat_dist[idx[i]])
            if acc[i] >= 0:
                avg_acc = avg_acc + acc[i]
                cnt += 1
                visible[i] = 1
            else:
                acc[i] = 0
        avg_acc = avg_acc / cnt if cnt != 0 else 0

    # PCKh
    PCKh = np.zeros((len(idx)))
    avg_PCKh = 0

    if dataset == "Penn_Action":
        neck = [(target[0, 1, 0]+target[0, 2, 0])/2,
                (target[0, 1, 1]+target[0, 2, 1])/2]
        headLength = np.linalg.norm(target[0, 0, :] - neck)
    elif dataset == "Sub-JHMDB":
        headLength = 2*(np.linalg.norm(target[0, 0, :] - target[0, 2, :]))

    for i in range(len(idx)):
        PCKh[i] = dist_acc(dists[idx[i]], thr_PCKh*headLength)
        if PCKh[i] >= 0:
            avg_PCKh = avg_PCKh + PCKh[i]
        else:
            PCKh[i] = 0

    avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

    PCK = np.zeros((len(idx)))
    avg_PCK = 0

    if dataset == "Penn_Action":
        if normTorso:
            torso = np.linalg.norm(target[0, 2,:]-target[0, 7,:])
            if torso < 1:
            	torso = np.linalg.norm(pred[0, 2,:]-pred[0, 7,:])
        else:
            torso = max(abs(bbox[0, 2, 0] - bbox[0, 0, 0]), abs(bbox[0, 2, 1] - bbox[0, 0, 1]))
        
        for i in range(len(idx)):
            PCK[i] = dist_acc(dists[idx[i]], thr_PCK*torso)

            if PCK[i] >= 0:
                avg_PCK = avg_PCK + PCK[i]
            else:
                PCK[i] = 0
        avg_PCK = avg_PCK / cnt if cnt != 0 else 0

    elif dataset == "Sub-JHMDB":
        if normTorso:
            torso = np.linalg.norm(target[0, 4,:]-target[0, 5,:])
            if torso < 1:
                torso = np.linalg.norm(pred[0, 4,:]-pred[0, 5,:])
        else:
            torso = threshold * max(abs(bbox[0, 2, 0] - bbox[0, 0, 0]), abs(bbox[0, 2, 1] - bbox[0, 0, 1]))
    
        for i in range(len(idx)):
            # if i != 0 or i != 1:
            PCK[i] = dist_acc(dists[idx[i]], thr_PCK*torso)

            if PCK[i] >= 0:
                avg_PCK = avg_PCK + PCK[i]
            else:
                PCK[i] = 0
        avg_PCK = avg_PCK / cnt if cnt != 0 else 0

    return acc, PCK, PCKh, cnt, pred, visible


def get_preds(heatmaps):
    if heatmaps.dim() != 4:
        raise ValueError('Input must be 4-D tensor')
    max_val, max_idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), heatmaps.size(2) * heatmaps.size(3)), 2)
    preds = torch.Tensor(max_idx.size(0), max_idx.size(1), 2)
    preds[:, :, 0] = max_idx[:, :] % heatmaps.size(3)
    preds[:, :, 1] = max_idx[:, :] // heatmaps.size(3)
    preds[:, :, 1] = preds[:, :, 1].floor()
    return preds

def calc_train_dists(preds, labels, normalize):
    dists = torch.Tensor(preds.size(1), preds.size(0))
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            if labels[i, j, 0] == 0 and labels[i, j, 1] == 0:
                dists[j, i] = -1
            else:
                dists[j, i] = torch.dist(labels[i, j, :], preds[i, j, :]) / normalize
    return dists

def dist_accuracy(dists, th=0.2):
    if torch.ne(dists, -1).sum() > 0:
        return (dists.le(th).eq(dists.ne(-1)).sum()) * 1.0 / dists.ne(-1).sum()
    else:
        return -1

def cal_train_acc(output, target):
    
    b, t, c, h, w = output.size()
    num_of_joints = c

    output = output.view(-1, c, h, w)
    target = output.view(-1, c, h, w)

    preds = get_preds(output)
    gt = get_preds(target)
    dists = calc_train_dists(preds, gt, output.size(3) / 10.0)

    avg_acc = 0.0
    bad_idx_count = 0
    for ji in range(num_of_joints):
        acc = dist_accuracy(dists[ji, :])
        if acc > 0:
            avg_acc += acc
        else:
            bad_idx_count += 1
    if bad_idx_count != num_of_joints:
        avg_acc = avg_acc / (num_of_joints - bad_idx_count)
    return avg_acc


def get_PCKh_jhmdb(Test_gt, Test_out, Bbox,imgPath ,normTorso):
    from prettytable import PrettyTable
    # 0: neck    1:belly   2: face
    # 3: right shoulder  4: left shoulder
    # 5: right hip       6: left hip
    # 7: right elbow     8: left elbow
    # 9: right knee      10: left knee
    # 11: right wrist    12: left wrist
    # 13: right ankle    14: left ankle

    orderJHMDB = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    thresh = 0.2
    N = Test_out.shape[1]
    if normTorso:
        torso_norm = 1 # 1: Torso / 0:bbx; default as 0 -> 0.2*max(h,w)
    else:
        torso_norm = 0
    sample_num = Test_gt.shape[0]

    HitPoint = np.zeros((sample_num, len(orderJHMDB)))
    allPoint = np.ones((sample_num,  N, len(orderJHMDB)))
    Point_to_use = np.ones((sample_num, len(orderJHMDB)))


    for sample in range(0, sample_num):
        # print('test sample:', sample)
        test_gt = Test_gt[sample]
        test_out = Test_out[sample]
        nframes = Test_gt[sample].shape[0]
        img_path = imgPath[sample]
        bbox = Bbox[sample]

        # num_frame = test_gt.shape[0]
        if nframes >= test_gt.shape[0]:
            nfr = test_gt.shape[0]
        else:
            nfr = nframes.int()

        seqError = torch.zeros(nfr, len(orderJHMDB))
        seqThresh = torch.zeros(nfr,  len(orderJHMDB))

        for frame in range(0, nfr):
            gt = test_gt[frame] # 13x2
            pred = test_out[frame] # 13x2
            # vis = visibility[frame] # 1x13

            if torso_norm == 1:
                bodysize = torch.norm(gt[4] - gt[5])
                if bodysize < 1:
                    bodysize = torch.norm(pred[4] - pred[5])
            else:
                bodysize = torch.max(bbox[frame, 2]-bbox[frame, 0], bbox[frame, 3] - bbox[frame, 1])

            error_dis = torch.norm(gt-pred, p=2, dim=1, keepdim=False)

            seqError[frame] = torch.FloatTensor(error_dis)
            # seqThresh[frame] = (bodysize * thresh) * torch.ones(partJHMDB)
            seqThresh[frame] = (bodysize*thresh) * torch.ones(len(orderJHMDB))

        pts = allPoint[sample, 0:nfr]
        Point_to_use[sample] = np.sum(pts, axis=0)

        less_than_thresh = seqError.numpy()<=seqThresh.numpy()
        HitPoint[sample] = np.sum(less_than_thresh, axis=0)
        eachPCK = np.divide(np.sum(HitPoint[sample], axis=0), np.sum(Point_to_use[sample], axis=0))
        eachMean = np.mean(eachPCK)

        # print('sample num:', sample, 'imgpath:', img_path, 'eachMean:', eachMean)


    finalPCK = np.divide(np.sum(HitPoint, axis=0), np.sum(Point_to_use, axis=0))

    finalMean = np.mean(finalPCK)

    mPCKtable = PrettyTable(['Joints', 'mPCK'])
    mPCKtable.add_row(['Head', finalPCK[2]])
    mPCKtable.add_row(['Shoulder', 0.5*(finalPCK[3]+finalPCK[4])])
    mPCKtable.add_row(['Elbow', 0.5*(finalPCK[7]+finalPCK[8])])
    mPCKtable.add_row(['Wrist', 0.5*(finalPCK[11]+finalPCK[12])])
    mPCKtable.add_row(['Hip', 0.5*(finalPCK[5]+finalPCK[6])])
    mPCKtable.add_row(['Knee', 0.5*(finalPCK[9]+finalPCK[10])])
    mPCKtable.add_row(['Ankle', 0.5*(finalPCK[13]+finalPCK[14])])
    mPCKtable.add_row(['Mean', finalMean])
    print(mPCKtable)

    return finalMean, finalPCK