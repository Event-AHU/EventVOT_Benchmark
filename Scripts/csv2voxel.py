import os
import pdb
import csv
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import scipy.io
from spconv.pytorch.utils import PointToVoxel
# from dv import AedatFile
import numpy as np


def transform_points_to_voxels(data_dict={}, voxel_generator=None, device=torch.device("cpu")):
    """
    将点云转换为voxel,调用spconv的VoxelGeneratorV2
    """
    points = data_dict['points']
    # 将points打乱
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    data_dict['points'] = points

    # 使用spconv生成voxel输出
    points = torch.as_tensor(data_dict['points']).to(device)
    voxel_output = voxel_generator(points)

    # 假设一份点云数据是N*4，那么经过pillar生成后会得到三份数据
    # voxels代表了每个生成的voxel数据，维度是[M, 5, 4]
    # coordinates代表了每个生成的voxel所在的zyx轴坐标，维度是[M,3]
    # num_points代表了每个生成的voxel中有多少个有效的点维度是[m,]，因为不满5会被0填充
    voxels, coordinates, num_points = voxel_output
    voxels = voxels.to(device)
    coordinates = coordinates.to(device)
    num_points = num_points.to(device)
    # 选event数量在前5000的voxel  8000 from(4k+,6k+)
    # print(torch.where(num_points>=16)[0].shape)
    if num_points.shape[0] < 4000:
        features = voxels[:, :, 3]
        coor = coordinates[:, :]
    else:
        _, voxels_idx = torch.topk(num_points, 4000)
        # 将每个voxel的1024个p拼接作为voxel初始特征
        features = voxels[voxels_idx][:, :, 3]
        # 前5000个voxel的三维坐标
        coor = coordinates[voxels_idx]
    # 将y.x.t改为t,x,y
    coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]

    return coor, features


if __name__ == '__main__':
    device = torch.device("cpu")
    data_path = ''
    save_path = ''
    video_files = os.listdir(data_path)
    dvs_img_interval = 1
    voxel_generator = PointToVoxel(
        # 给定每个voxel的长宽高  [0.05, 0.05, 0.1]
        vsize_xyz=[50, 17, 13],  # [0.2, 0.25, 0.16]  # [15, 17, 10]  [30, 35, 26]        # 因此坐标范围（9,9,9）
        # 给定点云的范围 [  0.  -40.   -3.   70.4  40.    1. ]
        coors_range_xyz=[0, 0, 0, 1000, 345, 259],
        # 给定每个点云的特征维度，这里是x，y，z，r 其中r是激光雷达反射强度       # 346x260  t,x,y
        num_point_features=4,
        # 最多选取多少个voxel，训练16000，推理40000
        max_num_voxels=10000,  # 16000
        # 给定每个pillar中有采样多少个点，不够则补0  因此我将neg voxel改为-1;
        max_num_points_per_voxel=16,  # 1024
        device=device
    )
    for videoID in tqdm(range(len(video_files))):
        fileName = video_files[videoID]
        foldName = os.path.splitext(fileName)[0]
        print("============>> foldName: ", foldName)
        voxel_path = os.path.join(save_path,  foldName+"_voxel/")
        mat_save = os.path.join(voxel_path)
        if not os.path.exists(voxel_path):
            os.makedirs(os.path.join(voxel_path))
        # for FileID in tqdm(range(len(fileLIST))):
            # videoName = fileLIST[FileID]
            # (filename, extension) = os.path.splitext(videoName)
            # # print("==>> filename: ", filename)
            # if (extension == '.csv'):
        voxel = os.path.join(voxel_path,"frame0498.mat")
        if os.path.exists(voxel):
            print("Skip this video : ", fileName)
            continue
        read_path = os.path.join(data_path, fileName)

        ## read csv;
        dt = pd.read_csv(read_path, dtype=np.int32, delimiter=",", usecols=(0, 1, 2, 3) )
        dt = np.array(dt)
        dt = torch.tensor(dt, dtype=torch.int)
        x_all, y_all, p_all, t_all = torch.chunk(dt, 4, dim=1)
        all_events = torch.cat(( x_all, y_all, p_all, t_all), dim=1)
        all_events = all_events.numpy()

        time_length = all_events[-1,3] - all_events[0,3]
        eventMODnum = 50000
        frameNUM = int(5000000//33333)
        start_idx = []
        # deltaT = 33333	#finalTIME_stamp//33333
        deltaT = int(time_length/499)
        i = 0
        for j in range(len(all_events)):
            if (all_events[j][-1]-all_events[0][-1]) >= (deltaT * i):
                start_idx.append(j)
                i += 1
                
        start_time_stamp = 0
        saved_event_timeStamp = []
        count_IMG = 0
        
        for imgID in range(len(start_idx)-1):
            start_time_stamp = start_idx[imgID]
            end_time_stamp = start_idx[imgID+1]

            saved_event_timeStamp.append([start_time_stamp, end_time_stamp])
            event = all_events[start_time_stamp:end_time_stamp]
            t = t_all[start_time_stamp:end_time_stamp]

            time_length = t[-1] - t[0]
            ## rescale the timestampes to start from 0 up to 1000
            t = ((t-t[0]).float() / time_length) * 1000
            all_idx = np.where(event[:,2] != -1)      # all event
            neg_idx = np.where(event[:,2] == 0)       # neg event
            t = t[all_idx]
            x = torch.from_numpy(event[:,0][all_idx]).unsqueeze(-1)
            y = torch.from_numpy(event[:,1][all_idx]).unsqueeze(-1)
            p = torch.from_numpy(event[:,2][all_idx]).unsqueeze(-1)
            p[neg_idx] = -1     # negtive voxel change from 0 to -1. because after append 0 operation.
 
            current_events = torch.cat((t, x, y, p), dim=1)
            # if current_events.shape[0] < 3000:   # remove it
            #     continue
            data_dict = {'points': current_events}

            coor, features = transform_points_to_voxels(data_dict=data_dict, voxel_generator=voxel_generator,
                                                        device=device)
            coor = coor.cpu().numpy()
            features = features.cpu().numpy()
            # print('coor', coor)
            scipy.io.savemat(mat_save + 'frame{:0>4d}.mat'.format(count_IMG), mdict={'coor': coor, 'features': features})   # coor: Nx(t,x,y);   features:Nx32 or Nx10024
            count_IMG += 1