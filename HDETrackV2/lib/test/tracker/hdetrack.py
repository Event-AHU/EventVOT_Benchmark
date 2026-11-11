import math
from PIL import Image
import numpy as np

from lib.models.hdetrack import build_hdetrack_s
from lib.test.tracker.basetracker import BaseTracker
import torch
import copy

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import torch.nn.functional as F
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from peft import LoraConfig, get_peft_model, AdaLoraConfig
from .show_CAM import getCAM2

import random
from torch.optim.lr_scheduler import StepLR,MultiStepLR
from ...utils.heapmap_utils import generate_heatmap
from ...utils.focal_loss import FocalLoss
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.box_ops import giou_loss

class HDETrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(HDETrack, self).__init__(params)
        init_seeds(3407)
        network = build_hdetrack_s(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        print("Load checkpoints from:", self.params.checkpoint)
        self.cfg = params.cfg
        self.network = network.cuda()
        # self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
                
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        
        self.lora = LoraConfig(  
            r=32,
            lora_alpha=64,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"], 
            lora_dropout=0.01,
            )  
        
        self.network = get_peft_model(self.network, self.lora)
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(parameters, lr=0.01)
        
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[4, 8, 16], gamma=0.1, last_epoch=-1)
        self.start_train_id = 6
        self.epoch_num = 5
        self.gaussian_maps_list = []
        self.search_list = []
        self.focal_loss = FocalLoss()
        self.failed_search_num = 0
        self.pre_box = None
        
    # def initialize(self, image, start_frame_idx,  info: dict):
    def initialize(self, image, start_frame_idx, img_templates, info: dict):
        ## forward the template once
        z_patch_arr, resize_factor, z_amask_arr, crop_coor = sample_target(image, info['init_bbox'],
                                                                           self.params.template_factor,
                                                                           output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
            
        ## for other templates
        self.template_list = []
        for imgs_i in img_templates:
            z_patch_arr_i, resize_factor_i, z_amask_arr_i, crop_coor_i = sample_target(imgs_i, info['init_bbox'],
                                                                        self.params.template_factor,
                                                                        output_sz=self.params.template_size)
            self.z_patch_arr_i = z_patch_arr_i
            template_i = self.preprocessor.process(z_patch_arr_i, z_amask_arr_i)
            self.template_list.append(template_i)
            
        self.state = info['init_bbox']
        self.frame_id = start_frame_idx
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    # def track(self, image, info: dict = None):
    def track(self, image, search_num, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        if self.failed_search_num == 7:
            print("Search from global image...")
            print(self.frame_id)
            # state = [0, 0, W, H]
            x_patch_arr, resize_factor, x_amask_arr, crop_coor = sample_target(image, self.state, self.params.search_factor*1.5,
                                                                    output_sz=self.params.search_size)   # (x1, y1, w, h)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)
            search = search.tensors

            self.failed_search_num = 0
        else: 
            x_patch_arr, resize_factor, x_amask_arr, crop_coor = sample_target(image, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)   # (x1, y1, w, h)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)
            search = search.tensors

        if search_num == self.start_train_id:
            self.network.train()
            for epoch in range(self.epoch_num):
                self.optimizer.zero_grad() 
                pred_score_map_list = []
                for i in range(len(self.search_list)):
                    x_dict = self.search_list[i]
                    out_dict = self.network.forward(template=self.z_dict1.tensors, 
                                                    search=x_dict, 
                                                    ce_template_mask=self.box_mask_z)
                    # add hann windows
                    pred_score_map = out_dict['s_score_map']
                    pred_score_map_list.append(pred_score_map)    

                x_dict = search
                out_dict = self.network.forward(template=self.z_dict1.tensors, 
                                                search=x_dict, 
                                                ce_template_mask=self.box_mask_z)
                # add hann windows
                pred_score_map = out_dict['s_score_map']
                response = self.output_window * pred_score_map
                
                response_list = []
                for template_i in self.template_list:
                    x_dict = search
                    out_dict_i = self.network.forward(template=template_i.tensors, 
                                                      search=x_dict,  
                                                      ce_template_mask=self.box_mask_z)
                    response_i = self.output_window * out_dict_i['s_score_map']
                    response_list.append(response_i)
                
                self.comput_ttt_loss(pred_score_map_list, self.gaussian_maps_list, response, response_list, epoch)
            
        self.network.eval()
        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward( 
                template=self.z_dict1.tensors, 
                search=x_dict, 
                ce_template_mask=self.box_mask_z)
            
        # # add hann windows
        pred_score_map = out_dict['s_score_map']
        response = self.output_window * pred_score_map
        getCAM2(response, image, self.frame_id)
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['s_size_map'], out_dict['s_offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        ## Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if search_num < self.start_train_id:
            self.search_list.append(search)
            search_bbox = self.transform_bbox_to_crop(self.state, resize_factor, search.device)
            gt_gaussian_maps = generate_heatmap(search_bbox, self.params.search_size, self.params.backbone_size)[-1].unsqueeze(1)
            self.gaussian_maps_list.append(gt_gaussian_maps)
            
        pred_boxes_vec = box_cxcywh_to_xyxy(out_dict['s_pred_boxes']).view(-1, 4)
        if self.pre_box != None:
            _, iou = giou_loss(pred_boxes_vec, self.pre_box)
            if iou < 0.5:
                self.failed_search_num += 1
            else:
                if self.failed_search_num > 0:
                    self.failed_search_num -= 1
                else:
                    self.failed_search_num = 0
        
        self.pre_box = pred_boxes_vec

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights
        
    def comput_ttt_loss(self, pred_score_map, gaussian_map, response, response_list, epoch):
        pred_score_map = torch.cat(pred_score_map, dim=0)
        gaussian_map = torch.cat(gaussian_map, dim=0)
        loss1 = self.focal_loss(pred_score_map, gaussian_map)

        loss2 = torch.tensor(0.0, device=response.device)
        for other_response in response_list:
            loss_i = F.mse_loss(other_response, response, reduction='mean') * 1000
            loss2 += loss_i
            
        loss = loss1 + loss2        
        loss.backward()  
        self.optimizer.step()
        print("epoch_{} ttt_loss:".format(epoch), loss.item())    
        self.lr_scheduler.step()


def get_tracker_class():
    return HDETrack


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False