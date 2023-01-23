import os
import cv2
import numpy as np

root_dir = 'data/Penn_Action'
frame_dir = os.path.join(root_dir, 'frames')
frame_name = '000001.jpg'
train_label_dir = os.path.join(root_dir, 'test')

train_videos = os.listdir(train_label_dir)

for i in range(len(train_videos)):
    img_path = os.path.join(frame_dir, train_videos[i].split('.')[0], frame_name)
    img = cv2.imread(img_path, 0)
    img_size = [img.shape[0], img.shape[1]]
    anno = np.load(os.path.join(train_label_dir, train_videos[i]), allow_pickle=True).item()
    dim = anno['dimensions']
    
    if img_size != dim:
        dim = img_size
    anno['dimensions'] = dim
    np.save(os.path.join(train_label_dir, train_videos[i]), anno)
print('Save Done~~')