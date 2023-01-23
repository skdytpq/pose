import os
import cv2
import glob
import numpy as np
import scipy.io as scio


def getBox(mask):
    x, y = mask.nonzero()
    bbox1 = np.min(x) - 1
    bbox2 = np.min(y) - 5
    bbox3 = np.max(x) + 1
    bbox4 = np.max(y) + 3
    
#     print(x)
#     print(y)
    bbox_h = bbox4 - bbox2
    bbox_w = bbox3 - bbox1
    
    bbox = [bbox2, bbox1, bbox4, bbox3]
    return bbox

root_dir = './data/Sub-JHMDB/'
split_dir = os.path.join(root_dir, 'sub-splits/')
puppet_dir = os.path.join(root_dir, 'puppet_mask/')
joint_position_dir = os.path.join(root_dir, 'joint_positions/')
image_dir = os.path.join(root_dir, 'Rename_Images/')
actions = os.listdir(puppet_dir)
# print(len(actions))

split = [1, 2, 3]
for subId in range(len(split)):
    print('Sub-script {}:'.format(subId + 1))
    fileName = '*split_' + str(subId + 1) + '.txt'
    fileInFolder = split_dir + fileName
    print(fileInFolder)
    subFiles = glob.glob(fileInFolder)
    print(len(subFiles))
    for i in range(len(subFiles)):
        fileName = subFiles[i].split('/')[-1]
        pos_end = fileName.find('_test')
        category = fileName[:pos_end]
        
        with open(os.path.join(split_dir, fileName), 'r') as f:
            fid = f.readlines()
        
        # Reading Videos
        for j in range(len(fid)):
            seqName = fid[j].split('.')[0]
            trainTest = float(fid[j][-2:])

            seqFolder = os.path.join(image_dir, category, seqName)
            annoName = joint_position_dir + '/' + category + '/'+ seqName+'/joint_positions.mat'
            anno = scio.loadmat(annoName)
            nframes = anno['pos_img'].shape[-1]
            imgSummary = glob.glob(seqFolder + '/*.png')
            if nframes != len(imgSummary):
                assert('Label length and frame length do not match...\n')
            image = []
            
            # Reading Frames
            for frame in range(nframes):
                try:
                    img = cv2.imread(imgSummary[frame])
                except:
                    print('Error in image reading...\n')
                image.append(img)
            
            # Get bounding box from mask label 
            maskName = puppet_dir + '/' + category + '/' + seqName + '/' + '/puppet_mask.mat'
            try:
                maskLabel = scio.loadmat(maskName)
                mask = maskLabel['part_mask']
#                 print(mask.shape, np.max(mask))
            except:
                assert('Can not find the bounding box for sequence... \n')
            
            bbox = np.zeros((nframes, 4))
            for frame in range(nframes):
                bbox[frame, :] = getBox(mask[:, :, frame])
            save_dir = puppet_dir + '/' + category + '/' + seqName + '/' + 'Bbox.mat'
            # print(save_dir)
            scio.savemat(save_dir, {'Bbox': bbox})