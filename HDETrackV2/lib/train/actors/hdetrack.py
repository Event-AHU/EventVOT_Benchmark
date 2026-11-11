from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

import torch.nn.functional as F
import torch.nn as nn

class HDETrackActor(BaseActor):
    """ Actor for training HDETrack models """
    def __init__(self, net_teacher, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net_teacher, net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict, out_dict_s = self.forward_pass(data)
        
        # compute losses
        loss, status = self.compute_losses(out_dict, out_dict_s, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 2 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == self.settings.num_search
        assert len(data['template_event']) == 1
        assert len(data['search_event']) == self.settings.num_search
        # assert len(data['template_event_images']) == 1
        # assert len(data['search_event_images']) == 1

        template_list = []
        search_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
            search_list.append(search_img_i)
        
        event_template_list = []
        event_search_list = []
        for i in range(self.settings.num_template):
            event_template_i = data['template_event'][i].view(-1, *data['template_event'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            event_template_list.append(event_template_i)
        for i in range(self.settings.num_search):
            event_search_i = data['search_event'][i].view(-1, *data['search_event'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            event_search_list.append(event_search_i)
        
        # template_event = data['template_event'][0].view(-1, *data['template_event'].shape[2:])
        # search_event = data['search_event'][0].view(-1, *data['search_event'].shape[2:])

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                    total_epochs=ce_start_epoch + ce_warm_epoch,
                                                    ITERS_PER_EPOCH=1,
                                                    base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
        if len(template_list) == 1:
            template_list = template_list[0]
        if len(event_template_list) == 1:
            event_template_list = event_template_list[0]

        out_dict = self.net_teacher(
                            template=template_list,
                            search=search_list,
                            event_template=event_template_list,
                            event_search=event_search_list,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
                                
        out_dict_s = self.net(
                            template=template_list,
                            search=search_list,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        
        return out_dict, out_dict_s

    def compute_losses(self, pred_dict, pred_dict_s, gt_dict, return_status=True):
        assert isinstance(pred_dict, list)
        assert isinstance(pred_dict_s, list)
        
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()
        total_giou = torch.tensor(0., dtype=torch.float).cuda()
        total_l1 = torch.tensor(0., dtype=torch.float).cuda()
        total_location = torch.tensor(0., dtype=torch.float).cuda()
        total_mse = torch.tensor(0., dtype=torch.float).cuda()
        total_l2 = torch.tensor(0., dtype=torch.float).cuda()
        total_response = torch.tensor(0., dtype=torch.float).cuda()
        total_mean_iou = torch.tensor([0], dtype=torch.float).cuda()
        
        # print(gt_dict['search_anno']) # (3, 16, 4)
        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        
        for i in range(len(pred_dict_s)):
            # get GT
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)
            # Get boxes
            pred_boxes = pred_dict_s[i]['s_pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
            # (B,4) --> (B,1,4) --> (B,N,4)

            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss
            total_giou += giou_loss
            
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            loss_dict['l1'] = l1_loss
            total_l1 += l1_loss
            
            # compute location loss
            if 's_score_map' in pred_dict_s[i]:
                location_loss = self.objective['focal'](pred_dict_s[i]['s_score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss
            total_location += location_loss
        
            temp = 2
            ################################# compute feature distilled loss ###########################################
            teacher_feature = pred_dict[i]['teacher_feature']       ## [bs, 640, 768]
            student_feature = pred_dict_s[i]['student_feature']     ## [bs, 320, 768]
            student_feature = student_feature.repeat(1,2,1)

            mse_loss =  F.mse_loss(student_feature, teacher_feature, reduction='mean')
            total_mse += mse_loss
            ################################## compute Similarity Matrix distilled loss ################################
            attn_teacher = pred_dict[i]['attn']        ## [bs, 12, 640, 640]
            attn_student = pred_dict_s[i]['s_attn']    ## [bs, 12, 320, 320]
            attn_student = attn_student.repeat(1,1,2,2)

            l2_loss = torch.mean(torch.nn.PairwiseDistance(p=2)(attn_student.float(), attn_teacher.float())) * 10
            total_l2 += l2_loss
            ################################ compute response distilled loss ###########################################
            response_t = pred_dict[i]['score_map']  / temp     ## [bs, 1, 16, 16]
            response_s = pred_dict_s[i]['s_score_map'] / temp  ## [bs, 1, 16, 16]

            response_loss = self.objective['focal'](response_s, response_t)
            total_response += response_loss
            
            ################################ compute fourier distilled loss ############################################
            if i == len(pred_dict_s) - 1:
                fft_loss = self.compute_fft_loss(pred_dict, pred_dict_s, response_t.shape[0]) * 10
                total_loss = total_loss + fft_loss
                
            # weighted sum
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + l2_loss + mse_loss + response_loss
            total_loss += loss
            
            mean_iou = iou.detach().mean()
            total_mean_iou += mean_iou
            
        if return_status:
            total_status = {"Loss/Total"     : total_loss.item(),
                            "Loss/Giou"      : total_giou.item(),
                            "Loss/L1"        : total_l1.item(),
                            "Loss/Location"  : total_location.item(),
                            "Loss/MSE"       : total_mse.item(),
                            "Loss/L2"        : total_l2.item(),
                            "Loss/Response"  : total_response.item(),
                            "Loss/FFT"       : fft_loss.item(),
                            "IoU"            : total_mean_iou.item()/len(pred_dict_s)
                            }
            return total_loss, total_status
        else:
            return total_loss
        
    def compute_fft_loss(self, pred_dict, pred_dict_s, B): 
        pred_t_list = []
        pred_s_list = []
        for i in range(len(pred_dict_s)):
            score_t = (pred_dict[i]['score_map']).view(B, -1)
            score_s = (pred_dict_s[i]['s_score_map']).view(B, -1)
            pred_t = (nn.functional.softmax((score_t), dim=-1)).view(B, 16, 16)
            pred_s = (nn.functional.softmax((score_s), dim=-1)).view(B, 16, 16)
            pred_t_list.append(pred_t)
            pred_s_list.append(pred_s)
        P_t = torch.stack((pred_t_list), dim=1)
        P_s = torch.stack((pred_s_list), dim=1)  
        fft_t = torch.fft.fft(P_t, dim=1, norm='forward').real * 10
        fft_s = torch.fft.fft(P_s, dim=1, norm='forward').real * 10
    
        fft_t = torch.sigmoid(fft_t).detach()
        fft_s = torch.sigmoid(fft_s)
        
        t_min = fft_t.amin(dim=(-2, -1), keepdim=True)
        t_max = fft_t.amax(dim=(-2, -1), keepdim=True)
        fft_t = (fft_t - t_min) / (t_max - t_min).clamp_min(1e-8)
        
        fft_loss = self.objective['focal'](fft_s, fft_t)
        
        return fft_loss