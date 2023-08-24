import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.hdetrack.vit import vit_base_patch16_224
from lib.models.hdetrack.vit_stu import vit_base_patch16_224_ce_s, vit_large_patch16_224_ce_s
from lib.utils.box_ops import box_xyxy_to_cxcywh


class HDETrack_S(nn.Module):
    """ This is the base class for hdetrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone_s = transformer
        self.box_head = box_head
        
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # self.fc = nn.Linear(768,1536)

    def forward(self, 
                event_template_img: torch.Tensor,                # torch.Size([bs, 3, 128, 128])
                event_search_img: torch.Tensor,                  # torch.Size([bs, 3, 256, 256])
                # event_template: torch.Tensor,                  # torch.Size([bs, 1, 19, 1024])
                # event_search: torch.Tensor,                    # torch.Size([bs, 1, 19, 4096])
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        # before feeding into backbone, we need to concat four vectors, or two two concat
        x, s_attn = self.backbone_s(z=event_template_img, x=event_search_img,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last,s_attn,None)
        
        return out

    def forward_head(self, event_feature, s_attn, gt_score_map=None):
        """
        event_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        ## dual head   768+768)*256
        event_search = event_feature[:, -self.feat_len_s:]       # [bs, 256, 768]
        opt = (event_search.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {
                   's_pred_boxes': outputs_coord_new,
                   's_score_map': score_map_ctr,
                   's_size_map': size_map,
                   's_offset_map': offset_map,
                   'student_feature': event_feature,
                   's_attn': s_attn
                   }
            return out
        else:
            raise NotImplementedError

def build_hdetrack_s(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('CEUTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
        
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone_s = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone_s.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone_s = vit_base_patch16_224_ce_s(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        # hidden_dim = backbone_s.embed_dim*2
        hidden_dim = backbone_s.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone_s = vit_large_patch16_224_ce_s(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )
        hidden_dim = backbone_s.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    backbone_s.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    box_head = build_box_head(cfg, hidden_dim)

    # cheakpoint=torch.load(os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE_T))
    # box_head.load_state_dict(cheakpoint['net'],strict=False)

    model_s = HDETrack_S(
        backbone_s,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    return model_s