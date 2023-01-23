from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = ''	
config.LOG_DIR = ''
config.DATA_DIR = ''
config.GPUS = '0'
config.WORKERS = 8
config.PRINT_FREQ = 2

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.MODEL = edict()
config.MODEL.STYLE = 'pytorch'

# Pose Initializer settings
config.POSE_INIT = edict()
config.POSE_INIT.INIT_WEIGHTS = True
config.POSE_INIT.PRETRAINED = ''
config.POSE_INIT.NUM_LAYERS = 101
config.POSE_INIT.NUM_DECONV_LAYERS = 3
config.POSE_INIT.NUM_DECONV_FILTERS = [256, 256, 256]
config.POSE_INIT.NUM_DECONV_KERNELS = [4, 4, 4]
config.POSE_INIT.FINAL_CONV_KERNEL = 1
config.POSE_INIT.TARGET_TYPE = 'gaussian'
config.POSE_INIT.DECONV_WITH_BIAS = False
config.POSE_INIT.SIGMA = 2
config.POSE_INIT.CHECKPOINT = 'models/pytorch/mpii/penn/pose_resnet_{}/256x256_d256x3_adam_lr1e-3/model_best.pth.tar'.format(str(config.POSE_INIT.NUM_LAYERS))
config.POSE_INIT.HMDB_CHECKPOINT = 'models/pytorch/mpii/JHMDB/15-kps/pose_resnet_{}/256x256_d256x3_adam_lr1e-3/model_best.pth.tar'.format(str(config.POSE_INIT.NUM_LAYERS))

# Feature Extractor settings
config.FEA_EXT = edict()
config.FEA_EXT.INIT_WEIGHTS = True
config.FEA_EXT.PRETRAINED = ''
config.FEA_EXT.NUM_LAYERS = 50
config.FEA_EXT.NUM_DECONV_LAYERS = 3
config.FEA_EXT.NUM_DECONV_FILTERS = [256, 256, 256]
config.FEA_EXT.NUM_DECONV_KERNELS = [4, 4, 4]
config.FEA_EXT.FINAL_CONV_KERNEL = 1
config.FEA_EXT.TARGET_TYPE = 'gaussian'
config.FEA_EXT.DECONV_WITH_BIAS = False
config.FEA_EXT.SIGMA = 2
config.FEA_EXT.CHECKPOINT = 'models/pytorch/mpii/penn/pose_resnet_{}/256x256_d256x3_adam_lr1e-3/model_best.pth.tar'.format(str(config.FEA_EXT.NUM_LAYERS))
config.FEA_EXT.HMDB_CHECKPOINT = 'models/pytorch/mpii/JHMDB/pose_resnet_{}/256x256_d256x3_adam_lr1e-3/model_best.pth.tar'.format(str(config.FEA_EXT.NUM_LAYERS))


# Dataset Infor
config.DATASET = edict()
config.DATASET.DATASET = 'penn'
config.DATASET.ROOT = '/data/Penn_Action/'
config.DATASET.NUM_JOINTS = 13
config.DATASET.IMAGE_SIZE = [256, 256]
config.DATASET.HEATMAP_SIZE = [64, 64]
config.DATASET.FLIP = True

# Dataset Infor
config.JHMDB = edict()
config.JHMDB.DATASET = 'sub_jhmdb'
config.JHMDB.ROOT = '/data/Sub-JHMDB/'
config.JHMDB.NUM_JOINTS = 15
config.JHMDB.IMAGE_SIZE = [256, 256]
config.JHMDB.HEATMAP_SIZE = [64, 64]
config.JHMDB.FLIP = True

# DKD Training settings
config.TRAIN = edict()
config.TRAIN.OPTIMIZER = 'rms'
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [15, 25]
config.TRAIN.LR = 0.001 # 0.0005
config.TRAIN.SEQ = 5
config.TRAIN.BATCH_SIZE = 8
config.TRAIN.RESUME = True
config.TRAIN.SHUFFLE = True
config.TRAIN.ROT_FACTOR = 40
config.TRAIN.FLIP_CONTROL = 0.5

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True

# DKD Testing settings
config.TEST = edict()
config.TEST.BATCH_SIZE = 8
config.TEST.FLIP_TEST = True
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True
config.TEST.MODEL_FILE = ''

# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0
