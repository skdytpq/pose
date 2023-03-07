import numpy as np 
from scipy import io
import os
import pickle
import pandas as pd
base = '../data/labels'
whole_cat = ['strum_guitar' 'bench_press' 'jumping_jacks' 'tennis_serve' 'squat'
 'golf_swing' 'pullup' 'baseball_swing' 'tennis_forehand' 'bowl' 'pushup'
 'baseball_pitch' 'jump_rope' 'clean_and_jerk' 'situp']
cat = 'bench_press','squat','pullup','jumping_jacks' # ,'pushup','situp'
def read_mat():
    i = 0
    for name in os.listdir(base):
        i +=1
        mat_file = io.loadmat(os.path.join(base,name))
        mat_file = dict(mat_file)
        if mat_file['action'][0] in cat:
            
            name = f'{name.split(".")[0]}.npy'
            mat_file['framepath']  = name
            np.save(f'../data/npy_labels/{name}',np.array([dict(mat_file)]))
        else:
            pass

def split():
    i = 0
    file = []
    for name in os.listdir(base):
        mat_file = io.loadmat(os.path.join(base,name))
        mat_file = dict(mat_file)
        pose = mat_file['action'][0]
        data = {i : pose}
        file.append(data)
    file = pd.DataFrame(file)
    return file[0].unique()
if '__main__':
    read_mat()
#'__header__', '__version__', '__globals__', 'action', 'pose', 'x', 'y', 'visibility', 'train', 'bbox', 'dimensions', 'nframes'
#        nframes = mat_file['nframes']
#        dim = mat_file['dimensions']
#        x = mat_file['x']
#        y = mat_file['y']
#        visibility = mat_file['visibility']
#        bbox = mat_file['bbox']
#        img_path = os.path.join(base,name)
#        new.append([nframes , dim , x , y , visibility , bbox , img_path])
#        new = np.array(new)
#        new = new.reshape(-1,)