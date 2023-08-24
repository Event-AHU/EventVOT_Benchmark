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
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1
        assert len(data['template_event']) == 1
        assert len(data['search_event']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (bs, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:]) 
            template_list.append(template_img_i)  # (bs, 3, 128, 128)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (bs, 3, 256, 256)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:]) 

        template_event = data['template_event'][0].view(-1, *data['template_event'].shape[2:])
        search_event = data['search_event'][0].view(-1, *data['search_event'].shape[2:])

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

        out_dict = self.net_teacher(
                            template=template_list,
                            search=search_img,
                            event_template=template_event,
                            event_search=search_event,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
                                
        out_dict_s = self.net(
                            event_template_img=template_list,
                            event_search_img=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)                              
                            # event_template=template_event,
                            # event_search=search_event,
        
        return out_dict, out_dict_s

    def compute_losses(self, out_dict, out_dict_s, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        # Get boxes
        pred_boxes = out_dict_s['s_pred_boxes']
        
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                          max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        
        # compute location loss
        if 's_score_map' in out_dict_s:
            location_loss = self.objective['focal'](out_dict_s['s_score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # Distill loss
        temp = 2
        ################################# compute feature distilled loss ###########################################
        teacher_feature = out_dict['teacher_feature']       ## [bs, 640, 768]
        student_feature = out_dict_s['student_feature']     ## [bs, 320, 768]
        student_feature = student_feature.repeat(1,2,1)

        Mse_loss =  F.mse_loss(student_feature, teacher_feature, reduction='mean')  # * 1
        
        ################################## compute Similarity Matrix distilled loss ################################
        attn_teacher = out_dict['attn']        ## [bs, 12, 640, 640]
        attn_student = out_dict_s['s_attn']    ## [bs, 12, 320, 320]
        attn_student = attn_student.repeat(1,1,2,2)

        l2_loss = torch.mean(torch.nn.PairwiseDistance(p=2)(attn_student.float(), attn_teacher.float())) * 0.7 # * 10 

        ################################ compute response distilled loss ############################################
        response_t = out_dict['score_map']  / temp     ## [bs, 1, 16, 16]
        response_s = out_dict_s['s_score_map'] / temp  ## [bs, 1, 16, 16]

        response_loss = self.objective['focal'](response_s, response_t) * 0.7 # * 0.5
   
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + l2_loss + Mse_loss + response_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total"   : loss.item(),
                      "Loss/giou"    : giou_loss.item(),
                      "Loss/l1"      : l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/MSE"     : Mse_loss.item(),
                      "Loss/L2"      : l2_loss.item(),
                      "Loss/response": response_loss.item(),
                      "IoU"          : mean_iou.item()
                      }
            return loss, status
        else:
            return loss