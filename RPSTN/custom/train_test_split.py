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

def add_joint(file):
    try:
        file = np.load(file,allow_pickle = True)
        if file[0]['x'].shape[1] <14:
            for x_,y_ in zip(file[0]['x'],file[0]['y']):
                rev = [float((x_[9]+x_[10])/2),float((y_[9]+y_[10])/2)]
                spine = [float((x_[9]+x_[10] + x_[2] + x_[3])/4),float((y_[9]+y_[10]+y_[2] + y_[3])/4)]
                neck =  [float((x_[1]+x_[2] + x_[3])/3),float((y_[1]+y_[2]+y_[3] )/3)]
                np.append(file[0]['x'],rev[0])
                np.append(file[0]['x'],spine[0])
                np.append(file[0]['x'],neck[0])
                np.append(file[0]['y'],rev[1])
                np.append(file[0]['y'],spine[1])
                np.append(file[0]['y'],neck[1])
        else:
            pass
    except:
        pass
        # num_joint = 16
def joint_in(base):
    for f in os.listdir(base):
        ph = os.path.join(base,f)
        add_joint(ph)


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