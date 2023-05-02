import numpy as np
import os
import shutil

def move_visible(files,i,path):
    name = []
    if 'npy' in files:
        file = np.load(path,allow_pickle = True).copy() # 
        if  0 in file[0]['x'] or 0 in file[0]['x']:
            name.append(files)
    for fil in name:
        if i =='train':
            shutil.move(f'data/pose_data/{i}/{fil}',f'data/pose_data/itedata/{i}/{fil}')
        if i =='val':
            shutil.move(f'data/pose_data/{i}/{fil}',f'data/pose_data/itedata/{i}/{fil}')

for i in ['train','test']:
    lis = os.path.join(f'data/pose_data/{i}')
    for file in os.listdir(lis):
        path = os.path.join(lis,file)
        move_visible(file,i,path)


           