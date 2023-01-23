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
    
def moving(train,test):
     for i in os.listdir(base):
        file = np.load(os.path.join(base,i),allow_pickle = True)
        if file[0]['framepath'] in  train:
            shutil.move(os.path.join(base,i),os.path.join(base_train,i))
        elif file[0]['framepath'] in test:
            shutil.move(os.path.join(base,i),os.path.join(base_test,i))
        else:
            print('ERROR : exception occured! please check y_data category')
            break
if '__main__':
    split(base)
    train,test = stratify()
    moving(train,test)
    split(test)
    print('=====================Ratio of data & Ratio of train======================')