import sys
sys.path.append("ITES")
sys.path.append("ITES/common")
sys.path.append("ITES/data")
sys.path.append("ITES/img")
sys.path.append("ITES/checkpoint")
sys.path.append('RPSTN/custom')
sys.path.append('RPSTN/files')
sys.path.append('RPSTN/lib')
sys.path.append('RPSTN/pose_estimation')

from ITES import train_student
#from RPSTN.pose_estimation import train_penn

#model_pos_train = Teacher_net(poses_valid_2d[0].shape[-2],dataset.skeleton().num_joints(),poses_valid_2d[0].shape[-1],
#                            n_fully_connected=args.n_fully_connected, n_layers=args.n_layers, 
#                            dict_basis_size=args.dict_basis_size, weight_init_std = args.weight_init_std)

if '__main__':
    print(sys.path)