"""
Basic HIPTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from thop import profile
from thop.utils import clever_format
from lib.models.layers.head import build_box_head
from lib.models.hiptrack.vit import vit_base_patch16_224
from lib.models.hiptrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
import numpy as np
import cv2
import random
from lib.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, visualizeDuringTraining
import torchvision.ops as ops
from lib.models.hip import HistoricalPromptNetwork
from lib.models.hip.modules import KeyProjection
from lib.models.hip import ResBlock
import thop

class HIPTrack(nn.Module):
    """ This is the base class for HIPTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", vis_during_train=False, new_hip=False, memory_max=150, update_interval=20):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.HIP = HistoricalPromptNetwork()
        self.key_proj = KeyProjection(768, keydim=64)
        self.key_comp = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        self.searchRegionFusion = ResBlock(768, 768)
        self.new_hip = new_hip
        self.update_interval = update_interval
        if self.new_hip:
            self.upsample = nn.Upsample(scale_factor=2.0, align_corners=True, mode="bilinear")
        self.memorys = []
        self.mem_max = memory_max

    def set_eval(self):
        self.HIP.set_eval(mem_max=self.mem_max)

    def forward_track(self, index: int, template: torch.Tensor, template_boxes: torch.Tensor, search: torch.Tensor, ce_template_mask=None, ce_keep_rate=None, searchRegionImg=None, info=None):
        #self.HIP.memory.setPath("./visualizeMemBank", info, index)
        if index <= 10:
            x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate)

            B, _, Ht, Wt = template.shape
            _, _, C = x.shape
            _, _, Hs, Ws = search.shape
            
            upsampled_template = self.upsample(template)
            template_mask = self.generateMask([None, None, None], 
                                              template_boxes, 
                                              upsampled_template, x, 
                                              visualizeMask=False, cxcywh=False)

            template_feature = x[:, :(Ht // 16)**2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)

            template_feature = self.upsample(template_feature)

            ref_v_template = self.HIP('encode', 
                            upsampled_template, 
                            template_feature, 
                            template_mask.unsqueeze(1)) 
            
            k16_template = self.key_proj(template_feature)

            searchRegionFeature_1 = x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
            k16 = self.key_proj(searchRegionFeature_1)         

            searchRegionFeature_1_thin = self.key_comp(searchRegionFeature_1)

            historical_prompt = self.HIP('train_decode', 
                            k16, # queryFrame_key
                            searchRegionFeature_1_thin, # queryFrame value
                            k16_template.unsqueeze(2), # memoryKey
                            ref_v_template) # memoryValue

            B, C, H, W = historical_prompt.shape

            historical_prompt = historical_prompt.view(B, C, H*W).permute(0, 2, 1)

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            out = self.forward_head(
                torch.stack([feat_last[:, -self.feat_len_s:], historical_prompt], dim=0), 
                None, return_topk_boxes=False)
            
            out.update(aux_dict)
            out['backbone_feat'] = x

            if index == 5 or index == 10:
                #print(self.memorys)
                B, _, Ht, Wt = template.shape
                _, _, C = x.shape
                _, _, Hs, Ws = search.shape

                mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search, x, visualizeMask=False, frame=index, seqName=info)

                ref_v = self.HIP('encode', 
                                    search, 
                                    searchRegionFeature_1, 
                                    mask.unsqueeze(1))

                self.HIP.addMemory(k16, ref_v, searchRegionImg)
            return out

        else:
            #flops1, params1 = thop.profile(self.backbone, inputs=(template, search, ce_template_mask, ce_keep_rate, None, False, None, None))
            x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate)

            B, _, Ht, Wt = template.shape
            _, _, C = x.shape
            _, _, Hs, Ws = search.shape
            
            k16 = self.key_proj(x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))
            #flops2, params2 = thop.profile(self.key_proj, inputs=(x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16),))
            

            searchRegionFeature = x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
            searchRegionFeature_thin = self.key_comp(searchRegionFeature)
            #flops3, params3 = thop.profile(self.key_comp, inputs=(searchRegionFeature,))

            historicalPrompt = self.HIP('eval_decode', 
                            k16, # queryFrame_key
                            searchRegionFeature_thin, #queryFrame_value
                        )
            
            B, C, H, W = historicalPrompt.shape

            historicalPrompt = historicalPrompt.view(B, C, H*W).permute(0, 2, 1)

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            out = self.forward_head(
                torch.stack([feat_last[:, -self.feat_len_s:], historicalPrompt], dim=0), 
                None, return_topk_boxes=False) 

            out.update(aux_dict)
            out['backbone_feat'] = x

            if index % self.update_interval == 0:
                mask = self.generateMask(aux_dict['removed_indexes_s'], 
                                         out['pred_boxes'].squeeze(1), 
                                         search, x, visualizeMask=False, frame=index, seqName=info)

                ref_v = self.HIP('encode', 
                                    search, 
                                    searchRegionFeature, 
                                    mask.unsqueeze(1)) 
                
                self.HIP.addMemory(k16, ref_v, searchRegionImg)
            
            return out

    def forward(self, template: torch.Tensor,
                search: list,
                search_after: torch.Tensor=None,
                previous: torch.Tensor=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gtBoxes=None,
                previousBoxes=None,
                template_boxes=None
                ):
        '''
            template : [B 3 H_z W_z]
            search : [3 * [B 3 H_x W_x]]
            previous : [B L 3 H_x W_x]
        '''
        x, aux_dict = self.backbone(z=template, x=search[0],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        B, _, Ht, Wt = template.shape
        _, _, C = x.shape
        _, _, Hs, Ws = search[0].shape

        upsampled_template = self.upsample(template)
        template_mask = self.generateMask([None, None, None], 
                                          template_boxes.squeeze(0), 
                                          upsampled_template, x, 
                                          visualizeMask=False, cxcywh=False)
        template_feature = x[:, :(Ht // 16)**2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
        template_feature = self.upsample(template_feature)
        ref_v_template = self.HIP('encode', 
                        upsampled_template, 
                        template_feature, 
                        template_mask.unsqueeze(1)) 
        
        k16_template = self.key_proj(template_feature)
        searchRegionFeature_1 = x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        k16 = self.key_proj(searchRegionFeature_1)         
        searchRegionFeature_1_thin = self.key_comp(searchRegionFeature_1)
        historical_prompt = self.HIP('train_decode', 
                        k16, # queryFrame_key
                        searchRegionFeature_1_thin, # queryFrame value
                        k16_template.unsqueeze(2), # memoryKey
                        ref_v_template) # memoryValue
        B, C, H, W = historical_prompt.shape

        historical_prompt = historical_prompt.view(B, C, H*W).permute(0, 2, 1)
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(
            torch.stack([feat_last[:, -self.feat_len_s:], historical_prompt], dim=0), 
            None, return_topk_boxes=False) 
        out.update(aux_dict)
        out['backbone_feat'] = x

        mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search[0], x, visualizeMask=False)
        searchRegionFeature_1 = x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)

        
        ref_v = self.HIP('encode', 
                            search[0],
                            searchRegionFeature_1, 
                            mask.unsqueeze(1))

        k16 = self.key_proj(searchRegionFeature_1)
        #k16 = k16.reshape(B, *k16.shape[-3:]).transpose(1, 2).contiguous()

        x_2, aux_dict_2 = self.backbone(z=template, x=search[1],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        k16_2 = self.key_proj(x_2[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

        searchRegionFeature_2 = x_2[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        searchRegionFeature_2_thin = self.key_comp(searchRegionFeature_2)

        historicalPrompt_2 = self.HIP('train_decode', 
                            k16_2, # queryFrame_key
                            searchRegionFeature_2_thin, # queryFrame value
                            k16.unsqueeze(2), # memoryKey
                            ref_v) # memoryValue
        
        B, C, H, W = historicalPrompt_2.shape

        historicalPrompt_2 = historicalPrompt_2.view(B, C, H*W).permute(0, 2, 1)
        
        feat_x2_last = x_2
        if isinstance(x_2, list):
            feat_x2_last = x_2[-1]

        out_2 = self.forward_head(
            torch.stack([feat_x2_last[:, -self.feat_len_s:], historicalPrompt_2], dim=0), 
            None, return_topk_boxes=False)
        
        out_2.update(aux_dict_2)
        out_2['backbone_feat'] = x_2

        mask_2 = self.generateMask(aux_dict_2['removed_indexes_s'], 
                                 out_2['pred_boxes'].squeeze(1), 
                                 search[1], x_2, visualizeMask=False)

        ref_v_2 = self.HIP('encode', 
                            search[1], 
                            searchRegionFeature_2, 
                            mask_2.unsqueeze(1)) 
        
        x_3, aux_dict_3 = self.backbone(z=template, x=search[2],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        k16_3 = self.key_proj(x_3[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

        searchRegionFeature_3 = x_3[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        searchRegionFeature_3_thin = self.key_comp(searchRegionFeature_3)

        historicalPrompt_3 = self.HIP('train_decode', 
                            k16_3, # queryFrame_key
                            searchRegionFeature_3_thin, # queryFrame value
                            torch.cat([k16.unsqueeze(2), k16_2.unsqueeze(2)], dim=2), # memoryKey
                            torch.cat([ref_v, ref_v_2], dim=2)) # memoryValue

        historicalPrompt_3 = historicalPrompt_3.view(B, C, H*W).permute(0, 2, 1)

        feat_x3_last = x_3
        if isinstance(x_3, list):
            feat_x3_last = x_3[-1]

        out_3 = self.forward_head(
            torch.stack([feat_x3_last[:, -self.feat_len_s:], historicalPrompt_3], dim=0), 
            None, return_topk_boxes=False) 


        out_3.update(aux_dict_3)
        out_3['backbone_feat'] = x_3

        mask_3 = self.generateMask(aux_dict_3['removed_indexes_s'], 
                                 out_3['pred_boxes'].squeeze(1), 
                                 search[2], x_3, visualizeMask=False)

        ref_v_3 = self.HIP('encode', 
                            search[2], 
                            searchRegionFeature_3, 
                            mask_3.unsqueeze(1)) 

        x_4, aux_dict_4 = self.backbone(z=template, x=search[3],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        k16_4 = self.key_proj(x_4[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

        searchRegionFeature_4 = x_4[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        searchRegionFeature_4_thin = self.key_comp(searchRegionFeature_4)

        historicalPrompt_4 = self.HIP('train_decode', 
                            k16_4, # queryFrame_key
                            searchRegionFeature_4_thin, # queryFrame value
                            torch.cat([k16.unsqueeze(2), k16_2.unsqueeze(2), k16_3.unsqueeze(2)], dim=2), # memoryKey
                            torch.cat([ref_v, ref_v_2, ref_v_3], dim=2)) # memoryValue

        historicalPrompt_4 = historicalPrompt_4.view(B, C, H*W).permute(0, 2, 1)

        feat_x4_last = x_4
        if isinstance(x_4, list):
            feat_x4_last = x_4[-1]

        out_4 = self.forward_head(
            torch.stack([feat_x4_last[:, -self.feat_len_s:], historicalPrompt_4], dim=0), 
            None, return_topk_boxes=False) 

        out_4.update(aux_dict_4)
        out_4['backbone_feat'] = x_4

        mask_4 = self.generateMask(aux_dict_4['removed_indexes_s'], 
                                 out_4['pred_boxes'].squeeze(1), 
                                 search[3], x_4, visualizeMask=False)

        ref_v_4 = self.HIP('encode', 
                            search[3], 
                            searchRegionFeature_4, 
                            mask_4.unsqueeze(1))

        x_5, aux_dict_5 = self.backbone(z=template, x=search[4],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        k16_5 = self.key_proj(x_5[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

        searchRegionFeature_5 = x_5[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        searchRegionFeature_5_thin = self.key_comp(searchRegionFeature_5)

        historicalPrompt_5 = self.HIP('train_decode', 
                            k16_5, # queryFrame_key
                            searchRegionFeature_5_thin, # queryFrame value
                            torch.cat([k16.unsqueeze(2), k16_2.unsqueeze(2), k16_3.unsqueeze(2), k16_4.unsqueeze(2)], dim=2), # memoryKey
                            torch.cat([ref_v, ref_v_2, ref_v_3, ref_v_4], dim=2)) # memoryValue

        historicalPrompt_5 = historicalPrompt_5.view(B, C, H*W).permute(0, 2, 1)

        feat_x5_last = x_5
        if isinstance(x_5, list):
            feat_x5_last = x_5[-1]

        out_5 = self.forward_head(
            torch.stack([feat_x5_last[:, -self.feat_len_s:], historicalPrompt_5], dim=0), 
            None, return_topk_boxes=False) 

        out_5.update(aux_dict_5)
        out_5['backbone_feat'] = x_5

        return [out, out_2, out_3, out_4, out_5]

    def deNorm(self, image):
        img = image.cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img[0] = (img[0] * std[0] + mean[0]) * 255
        img[1] = (img[1] * std[1] + mean[1]) * 255
        img[2] = (img[2] * std[2] + mean[2]) * 255
        img = img.transpose(1, 2, 0).astype(np.uint8).copy()
        return img

    def generateMask(self, ceMasks, predBoxes, img_normed, img_feat, visualizeMask=False, cxcywh=True, frame=None, seqName=None):
        B, _, H_origin, W_origin = img_normed.shape
        masks = torch.zeros((B, H_origin, W_origin), device=img_feat.device, dtype=torch.uint8)
        pure_ce_masks = torch.ones((B, H_origin, W_origin), device=img_feat.device, dtype=torch.uint8)
        for i in range(B):
            if cxcywh:
                box = (box_cxcywh_to_xyxy((predBoxes[i])) * H_origin).int()
            else:
                box = (predBoxes[i] * H_origin).int()
                box[2] += box[0]
                box[3] += box[1]

            box[0] = 0 if box[0] < 0 else box[0]
            box[1] = H_origin if box[1] > H_origin else box[1]
            box[2] = W_origin if box[2] > W_origin else box[2]
            box[3] = 0 if box[3] < 0 else box[3]
            
            if visualizeMask:
                if not os.path.exists(f"./masks_vis/{seqName}/{frame}"):
                    os.makedirs(f"./masks_vis/{seqName}/{frame}")
                img = self.deNorm(img_normed[i])
            #masks[i] = torch.zeros((H_origin, W_origin), dtype=np.uint8)
            masks[i][box[1].item():box[3].item(), box[0].item():box[2].item()] = 1
            if ceMasks[0] is not None and ceMasks[1] is not None and ceMasks[2] is not None:
                ce1 = ceMasks[0][i]
                ce2 = ceMasks[1][i]
                ce3 = ceMasks[2][i]
                ce = torch.cat([ce1, ce2, ce3], axis=0)
                for num in ce:
                    x = int(num) // 24
                    y = int(num) % 24
                    masks[i][x*16 : (x+1)*16, y*16 : (y+1)*16] = 0
                    pure_ce_masks[i][x*16 : (x+1)*16, y*16 : (y+1)*16] = 0
            
            if visualizeMask:
                mask = masks[i].cpu().detach().numpy().astype(np.uint8)
                mask = np.stack([mask, mask, mask], axis=2) * 255
                pure_ce_mask = pure_ce_masks[i].cpu().detach().numpy().astype(np.uint8)
                pure_ce_mask = np.stack([pure_ce_mask, pure_ce_mask, pure_ce_mask], axis=2) * 255
                #cv2.rectangle(mask, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/mask.jpg", mask[:,:,::-1])
                #cv2.imwrite(f"./masks_vis/{seqName}/{frame}/ce_mask.jpg", pure_ce_mask[:,:,::-1])
                #import pdb; pdb.set_trace()
                img2 = img.copy()
                img3 = img.copy()
                img2[mask == 0] = 255
                img3[pure_ce_mask == 0] = 255
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img_with_mask.jpg", img2[:,:,::-1])
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img_CE_mask.jpg", img3[:,:,::-1])
                cv2.rectangle(img, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img.jpg", img[:,:,::-1])
        return masks

    def forward_head(self, cat_feature, gt_score_map=None, return_topk_boxes=False):
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        _, B, HW, C = cat_feature.shape
        H = int(HW ** 0.5)
        W = H
        originSearch = cat_feature[0].view(B, H, W, C).permute(0, 3, 1, 2)
        dynamicSearch = cat_feature[1].view(B, H, W, C).permute(0, 3, 1, 2)
        enc_opt = self.searchRegionFusion(originSearch + dynamicSearch).view(B, C, HW).permute(0, 2, 1)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
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
            score_map_ctr, bbox, size_map, offset_map, topkBbox = self.box_head(opt_feat, gt_score_map, return_topk_boxes)
            outputs_coord = bbox 
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            if return_topk_boxes:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                       'topk_pred_boxes': topkBbox,
                    }
            else:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                    }
            return out
        else:
            raise NotImplementedError


def build_hiptrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('HIPTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = HIPTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        new_hip=cfg.MODEL.NEW_HIP,
        memory_max=cfg.MODEL.MAX_MEM,
        update_interval=cfg.TEST.UPDATE_INTERVAL
    )

    if ('HIPTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models', cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
