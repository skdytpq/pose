import os
import scipy.io as sio
import oct2py

pose_directory = 'cdf'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
for subject in subjects:
    sub_path = os.path.join(f'cdf/{subject}', 'MyPoseFeatures/D3_Positions')
    dirs = os.listdir(os.path.join(f'cdf/{subject}', 'MyPoseFeatures/D3_Positions'))
    for filename in dirs:
        if filename.endswith('.cdf'):
            path = os.path.join(sub_path, filename)
            data = oct2py.cdfread(path)
            mat_filename = os.path.splitext(filename)[0] + '.mat'
            mat_path = os.path.join(dirs, mat_filename)
            sio.savemat(mat_path, {'data': data})