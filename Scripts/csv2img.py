import os 
import pdb 
import csv 
import numpy as np 
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

data_path = r'E:\Event_frame\uav'
save_path = r"E:\Event_frame\UAV"
#action_004_20220219_101957781_EI_70M`
if __name__=='__main__':
	cls_dirs = os.listdir(data_path)
	for cls_ID in range(len(cls_dirs)):
		cls = cls_dirs[cls_ID]
		# print(videoName[7:10])
		fileLIST = os.listdir(os.path.join(data_path))
		save_cls_path = os.path.join(save_path)
		if not os.path.exists(save_cls_path):
			os.makedirs(save_cls_path)
		for FileID in tqdm(range(len(fileLIST))):
			csv_Name = fileLIST[FileID]
			video_save_path = os.path.join(save_cls_path,csv_Name.split('.')[0])
			if os.path.exists(video_save_path):
				continue
			if not os.path.exists(video_save_path):
				os.makedirs(video_save_path)
			read_path = os.path.join(data_path,csv_Name)
			#action_003_20220215_172542477_0_E_100M.csv
			if '_E_' in csv_Name and len(csv_Name.split('_')[-3])>2:
				recordMODE = "E"
				dt = pd.read_csv(read_path, dtype=np.int32, delimiter=",",usecols=(0, 1, 2))  ## all_events.shape  (36267950, 5)
				dt = np.array(dt)
				dt = torch.tensor(dt, dtype=torch.int)
				p = torch.ones(dt.shape[0])
				p = p.unsqueeze(1)
				p = p.to(torch.int)
				y, x, t = torch.chunk(dt, 3, dim=1)
				all_events = torch.cat((x, y, p, t), dim=1)
			elif '_E_' in csv_Name and len(csv_Name.split('_')[-3])<=2:
				recordMODE = "E"
				dt = pd.read_csv(read_path, dtype=np.int32, delimiter=",", usecols=(0, 1, 2, 3) )
				dt = np.array(dt)
				dt = torch.tensor(dt, dtype=torch.int)
				x, y, p, t = torch.chunk(dt, 4, dim=1)
				all_events = torch.cat((x, y, p, t), dim=1)
			elif '_EI_' in csv_Name and len(csv_Name.split('_')[-3])>2:
				recordMODE = "EI"
				dt = pd.read_csv(read_path, dtype=np.int32, delimiter=",", usecols=(0, 1, 3, 4) )
				dt = np.array(dt)
				dt = torch.tensor(dt, dtype=torch.int)
				# dt = dt.to(device)
				y, x, p, t = torch.chunk(dt, 4, dim=1)
				mask = torch.where(p >= 0)
				t = t[mask].unsqueeze(1)
				x = x[mask].unsqueeze(1)
				y = y[mask].unsqueeze(1)
				p = p[mask].unsqueeze(1)
				all_events = torch.cat((x, y, p, t), dim=1)
			else:
				recordMODE = "EI"
				dt = pd.read_csv(read_path, dtype=np.int32, delimiter=",", usecols=(0, 1, 2, 3) )
				dt = np.array(dt)
				dt = torch.tensor(dt, dtype=torch.int)
				# breakpoint()
				x, y, p, t = torch.chunk(dt, 4, dim=1)
				all_events = torch.cat((x, y, p, t), dim=1)
			all_events = all_events.numpy()
			frameRATE = 30
			height,width = 720,1280
			finalTIME_stamp = int(all_events[all_events.shape[0]-1][3])
			# firstTIME_stamp = int(all_events[all_events.shape[0]][4])
			time_length = all_events[-1,3] - all_events[0,3]
			eventMODnum = 50000
			frameNUM = int(5000000//33333)
			start_idx = []
			# deltaT = 33333	#finalTIME_stamp//33333
			deltaT = int(time_length/500)
			i = 1
			for j in range(len(all_events)):
				if all_events[j][-1]-all_events[0][-1] > deltaT * i:
					start_idx.append(j)
					i += 1

			################################################################################
			###				save the Event Image
			################################################################################
			start_time_stamp = 0

			saved_event_timeStamp = []
			count_csvsample = 0
			count_IMG = 0
			assert len(start_idx)!=0,'{} get 0 img!'.format(csv_Name)
			# breakpoint()
			for imgID in range(len(start_idx)-1):
				event_frame = 255 * np.ones((height, width, 3), dtype=np.uint8)

				start_time_stamp = start_idx[imgID]
				end_time_stamp = start_idx[imgID+1]
                
				saved_event_timeStamp.append([start_time_stamp, end_time_stamp])
				if recordMODE == "E":
					event = all_events[start_time_stamp:end_time_stamp]
					event_frame[height - 1 - event[:, 1],  event[:, 0], :]  = [10, 10, 255] * 1
					cv2.imwrite(os.path.join(video_save_path, '{:04d}'.format(imgID+1)+'.png'), event_frame)
				else:
					event = all_events[start_time_stamp:end_time_stamp]
					on_idx = np.where(event[:, 2] == 1)
					off_idx = np.where(event[:, 2] == 0)
                    
					event_frame[height - 1 - event[:, 1][on_idx],  event[:, 0][on_idx], :] = [30, 30, 220] * event[:, 2][on_idx][:, None]
					event_frame[height - 1 - event[:, 1][off_idx], event[:, 0][off_idx], :] = [200, 30, 30] * (event[:, 2][off_idx]+1)[:, None]

					event_frame=cv2.flip(event_frame,0)  ##垂直翻转
					cv2.imwrite(os.path.join(video_save_path, '{:04d}'.format(count_IMG)+'.png'), event_frame)

				count_IMG += 1
