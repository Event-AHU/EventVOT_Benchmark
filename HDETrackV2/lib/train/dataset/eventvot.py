import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import scipy.io as scio

class EventVOT(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):

        root = env_settings().eventvot_dir if root is None else root
        super().__init__('eventvot', root, image_loader)

        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            if split == 'train':
                file_path = os.path.join(self.root, 'train.txt')
            elif split == 'val':
                file_path = os.path.join(self.root, 'val.txt')
            else:
                raise ValueError('Unknown split name')
            seq_ids = pandas.read_csv(file_path, header=None, dtype=np.int64).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))
         
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

    def get_name(self):
        return 'eventvot'

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id], "img")

    # def _get_event_img_sequence_path(self, seq_id):
    #     return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id] + "_dvs")

    def _get_grountgruth_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        bbox_path = self._get_grountgruth_path(seq_id)
        bbox = self._read_bb_anno(bbox_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        # return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
        return {'bbox': bbox, 'valid': valid, 'visible': visible, }

    def _get_frame_path(self, seq_path, frame_id):
        if os.path.exists(os.path.join(seq_path, '{:04}.png'.format(frame_id))):
            return os.path.join(seq_path, '{:04}.png'.format(frame_id))    # frames start from 0
        else:
            return os.path.join(seq_path, '{:04}.bmp'.format(frame_id))    # some image is bmp

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_event_sequence_path(self, seq_id):        ## get events' frames
        return os.path.join(self.root, self.sequence_list[seq_id], self.sequence_list[seq_id] + "_voxel")

    def _get_event_frame(self, seq_path, frame_id):
        frame_event_list = []
        for f_id in frame_id:
            event_frame_file = os.path.join(seq_path, 'frame{:04}.mat'.format(f_id))

            if os.path.getsize(event_frame_file) == 0:
                event_features = np.zeros(4096, 19)
                # need_data = [np.zeros([4096, 3]), np.zeros([4096, 16])]
            else:
                mat_data = scio.loadmat(event_frame_file)
                # need_data = [mat_data['coor'], mat_data['features']]
                event_features = np.concatenate((mat_data['coor'], mat_data['features']), axis=1)        # concat coorelate and features (x,y,z, feauture32/16)
                if np.isnan(event_features).any():
                    event_features = np.zeros(4096, 19)
                    print(event_frame_file, 'exist nan value in voxel.')
            frame_event_list.append(event_features)
        return frame_event_list

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        # obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]  ## Event_img
        # seq_event_path = self._get_event_img_sequence_path(seq_id)
        # frame_event_img_list = [self._get_frame(seq_event_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        
        seq_event_path = self._get_event_sequence_path(seq_id)
        frame_event_list = self._get_event_frame(seq_event_path, frame_ids)  ## Event_voxel

        return frame_list, anno_frames, object_meta, frame_event_list

