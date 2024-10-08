
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
from matplotlib import cm
import cv2
import pdb   

def draw_2d_pose(keypoints, skeleton, path):
    keypoints = np.asarray(keypoints.cpu())
    keypoints = keypoints - keypoints[0:1,:]
    keypoints[:,1:2] = -keypoints[:,1:2]
    # keypoints[:,0:1] = -keypoints[:,0:1]
    nkp = int(keypoints.shape[0])
    pid = np.linspace(0., 1., nkp)
    fig = plt.figure(1,figsize=(5,6))
    ax = fig.gca()
    # ax.set_xlim3d([-radius , radius])
    # ax.set_ylim3d([-radius, radius ])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    parents = skeleton.parents()
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'gray' if j in skeleton.joints_right() else 'orange'
        ax.plot([keypoints[j, 0], keypoints[j_parent, 0]],
                                    [keypoints[j, 1], keypoints[j_parent, 1]], linewidth=9,alpha=1,color=col)
    xs = keypoints[:,0]
    ys = keypoints[:,1]
    ax.scatter(xs, ys, s=80, c=pid, marker='o', cmap='gist_ncar',zorder=5)
    plt.axis('off')
    plt.savefig(path)
    plt.close()
    return
    
def draw_3d_pose(poses,image ,path,sub_path):
    #poses n*3 dataset.skeleton()
    poses = poses - poses[0:1,:]
    poses = np.asarray(poses.cpu())
    nkp = int(poses.shape[0])
    pid = np.linspace(0., 1., nkp)
    poses[:,1:2] = -poses[:,1:2]
    poses[:,0:1] = -poses[:,0:1]
    plt.ioff()
    fig = plt.figure()
    radius = np.abs(poses).max()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=15, azim=70)
    ax.set_xlim3d([-radius , radius])
    ax.set_zlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius ])
    # ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.patch.set_facecolor("white")  
    ax.dist = 7.5
    parents = [[1,16],[2,4],[3,5],[2,4],[3,5],[4,6],[5,7],[10,8],[9,11],[8,10],[9,11],[10,12],[11,13],[8,9],[14,16],[2,3]]
    parents = np.array(parents)-1
    x = poses[:,0]
    y = poses[:,1]
    z = poses[:,2]
    for bone in parents:
        ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]],[z[bone[0]],z[bone[1]]], 'r')
    xs = poses[:,0]
    zs = poses[:,1]
    ys = poses[:,2]
    #ax.scatter(xs, ys, zs, s=80, c=pid, marker='o', cmap='gist_ncar',zorder=2)
    # ax.scatter(xs, ys, zs, s=30, c='red', marker='o')
    plt.savefig(path,dpi=40)
    plt.close()
    resized_image = cv2.resize(image, 
                            (int(64), int(64)))
    plt.imshow(resized_image)
    plt.savefig(sub_path)
    return
# 0 -> 1,2
def draw_3d_pose1(poses, skeleton, path):
    #poses n*3 dataset.skeleton()
    poses = poses - poses[0:1,:]
    poses = np.asarray(poses.cpu())
    nkp = int(poses.shape[0])
    pid = np.linspace(0., 1., nkp)
    poses[:,1:2] = -poses[:,1:2]
    poses[:,0:1] = -poses[:,0:1]
    plt.ioff()
    fig = plt.figure()
    radius = np.abs(poses).max()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=15, azim=70)
    ax.set_xlim3d([-radius , radius])
    ax.set_zlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius ])
    # ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.patch.set_facecolor("white")  
    ax.dist = 7.5
    parents = skeleton.parents()
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'gray' if j in skeleton.joints_right() else 'orange'
        pos = poses

        ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 2], pos[j_parent, 2]],[pos[j, 1], pos[j_parent, 1]],linewidth=9,alpha=1,zdir='z', c=col)
    xs = poses[:,0]
    zs = poses[:,1]
    ys = poses[:,2]
    ax.scatter(xs, ys, zs, s=80, c=pid, marker='o', cmap='gist_ncar',zorder=2)
    # ax.scatter(xs, ys, zs, s=30, c='red', marker='o')
    plt.savefig(path,dpi=40)
    plt.close()
    return