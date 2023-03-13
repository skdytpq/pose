import os
import scipy.io as sio
import os
import cdflib
import numpy as np
import pdb
pose_directory = 'cdf'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
for subject in subjects:
    sub_path = os.path.join(f'monocdf/{subject}', 'MyPoseFeatures/D3_Positions_mono')
    sub_path2 = os.path.join(f'data/{subject}', 'MyPoseFeatures/D3_Positions_mono')
    os.makedirs(sub_path2,exist_ok=True)
    dirs = os.listdir(os.path.join(f'monocdf/{subject}', 'MyPoseFeatures/D3_Positions_mono'))
    for filename in dirs:
        if filename.endswith('.cdf'):
            path = os.path.join(sub_path, filename)
            data = cdflib.CDF(path)
            mat_filename = os.path.splitext(filename)[0]+'.cdf' + '.mat'
            mat_path = os.path.join(sub_path2, mat_filename)
            sio.savemat(mat_path, {'data': data[0]})    