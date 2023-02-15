import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
base = '../data/npy_labels'
base_train = '../data/train'
base_test = '../data/test'
test = '../data/train'
# 993개의 데이터 셋
def split(base):
    dat = os.listdir(base)
    category = []
    for i in os.listdir(base):
        file = np.load(os.path.join(base,i),allow_pickle = True)
        cat = file[0]['action']
        category.append(cat)
    df = pd.Series(category)
    print(df.value_counts())

#[squat]            231 
#[pushup]           211
#[pullup]           199
#[bench_press]      140
#[jumping_jacks]    112
#[situp]            100

def stratify():
    x_ = []
    y_ = []
    for i in os.listdir(base):
        file = np.load(os.path.join(base,i),allow_pickle = True)
        x = file[0]['framepath']
        y = file[0]['action']
        x_.append(x)
        y_.append(y)
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.3, random_state=777, stratify=y_) 
    return x_train,x_test

def add_joint(files):
    if 'npy' in files:
        file = np.load(files,allow_pickle = True).copy()
        if file[0]['x'].shape[1] <16:
            r,s,n = [],[],[]
            for x_,y_ in zip(file[0]['x'],file[0]['y']):
                rev = [float(round((x_[7]+x_[8])/2),3),float(round((y_[7]+y_[8])/2),3)]
                spine = [float(round((x_[7]+x_[8] + x_[1] + x_[2])/4),3),float(round((y_[7]+y_[8]+y_[1] + y_[2])/4,3))]
                neck =  [float(round((x_[0]+x_[1] + x_[2])/3),3),float(round((y_[0]+y_[1]+y_[2])/3),3)]
                r.append(rev)
                s.append(spine)
                n.append(neck)
            file[0]['x'] = np.concatenate((file[0]['x'],np.array(r)[:,0].reshape(-1,1)), axis=1)
            file[0]['x'] = np.concatenate((file[0]['x'],np.array(s)[:,0].reshape(-1,1)), axis=1)
            file[0]['x'] = np.concatenate((file[0]['x'],np.array(n)[:,0].reshape(-1,1)), axis=1)
            file[0]['y'] = np.concatenate((file[0]['y'],np.array(r)[:,1].reshape(-1,1)), axis=1)
            file[0]['y'] = np.concatenate((file[0]['y'],np.array(s)[:,1].reshape(-1,1)), axis=1)
            file[0]['y'] = np.concatenate((file[0]['y'],np.array(n)[:,1].reshape(-1,1)), axis=1)
            np.save(files,file)
        else:
            pass
    else:
        pass
def add_vis(files):
    if 'npy' in files:
        vl  = []
        file = np.load(files,allow_pickle = True).copy()
        for v in range(file[0]['visibility'].shape[0]):
            vl.append(np.array([1,1,1]))
        file[0]['visibility'] = np.concatenate((file[0]['visibility'],np.array(vl).reshape(-1,3)),axis= 1)
        # num_joint = 16

        np.save(files,file)
    else:
        pass
def joint_in(base):
    for f in os.listdir(base):
        ph = os.path.join(base,f)
        #add_joint(ph)
        add_vis(ph)

def moving(train,test):
     for i in os.listdir(base):
        file = np.load(os.path.join(base,i),allow_pickle = True)
        add_joint(file)
        if file[0]['framepath'] in  train:
            shutil.move(os.path.join(base,i),os.path.join(base_train,i))
        elif file[0]['framepath'] in test:
            shutil.move(os.path.join(base,i),os.path.join(base_test,i))
        else:
            print('ERROR : exception occured! please check y_data category')
            break
if '__main__':
    prepare = 0
    if prepare :
        split(base)
        train,test = stratify()
        moving(train,test)
        split(test)
    else:
        joint_in('../data/pose_data/train')
        joint_in('../data/pose_data/test')
    print('=====================Ratio of data & Ratio of train======================')