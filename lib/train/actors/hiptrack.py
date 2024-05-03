from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import numpy as np
import cv2

class HIPTrackActor(BaseActor):
    """ Actor for training HIPTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None, multiFrame=False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multiFrame = multiFrame

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
        out_dict = self.forward_pass(data)

        #import pdb
        #pdb.set_trace()
        if isinstance(out_dict, list):
            losses = None
            statuses = {"Loss/total": None,
                            "Loss/giou": None,
                            "Loss/l1": None,
                            "Loss/location": None,
                            "IoU": None
                            }
            for idx, out in enumerate(out_dict):
                partData = {"search_anno" : data['search_anno'][idx].unsqueeze(0)}
                loss, status = self.compute_losses(out, partData)
                if losses is None:
                    losses = loss
                else:
                    losses += loss
                
                for key, val in status.items():
                    if statuses[key] is None:
                        statuses[key] = val
                    else:
                        statuses[key] += val
            
            statuses['IoU'] = statuses['IoU'] / len(out_dict)
            return losses, statuses

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def deNorm(self, image):
        img = image.cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img[0] = (img[0] * std[0] + mean[0]) * 255
        img[1] = (img[1] * std[1] + mean[1]) * 255
        img[2] = (img[2] * std[2] + mean[2]) * 255
        img = img.transpose(1, 2, 0).astype(np.uint8).copy()
        cv2.imwrite("imgDeNorm.jpg", img=img[:,:,::-1])
        return img

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 5

        template_annos = data['template_anno']
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        previous_imgs = None
        previous_annos = None
        search_imgs = [data['search_images'][i].view(-1, *data['search_images'].shape[2:]) for i in range(len(data['search_images']))]  # (N_search, batch, 3, 384, 384)

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
        if self.multiFrame:
            out_dict = self.net(template=template_list,
                                search=search_imgs,
                                search_after=data['search_images'][1:],
                                previous=previous_imgs,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                gtBoxes=data['search_anno'],
                                previousBoxes=previous_annos,
                                template_boxes=template_annos)
        else:
            out_dict = self.net(template=template_list,
                                search=search_imgs,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                template_boxes=template_annos)
        #self.visualizeCE(out_dict['removed_indexes_s'], out_dict['pred_boxes'].squeeze(1), data['search_images'][0], data['search_anno'][0])
        return out_dict

    def visualizeCE(self, ceMasks, predBoxes, imgs, gtBoxes):
        import pdb
        pdb.set_trace()
        for i in range(16):
            mask = np.ones((24, 24), dtype=np.uint8)
            img = imgs[i]
            img = self.deNorm(img)
            ce1 = ceMasks[0][i]
            ce2 = ceMasks[1][i]
            ce3 = ceMasks[2][i]
            ce = torch.cat([ce1, ce2, ce3], axis=0)
            for num in ce:
                x = int(num) // 24
                y = int(num) % 24
                mask[x][y] = 0
            box = (box_cxcywh_to_xyxy((predBoxes[i])) * 384).int()
            gtBox = (box_xywh_to_xyxy(gtBoxes[i]) * 384).int()
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            mask = np.stack([mask, mask, mask], axis=2) * 255
            mask = cv2.resize(mask, (384, 384))
            cv2.rectangle(mask, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
            cv2.rectangle(img, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
            cv2.rectangle(mask, (gtBox[0].item(), gtBox[1].item()), (gtBox[2].item(), gtBox[3].item()), (0, 255, 0), 2)
            cv2.rectangle(img, (gtBox[0].item(), gtBox[1].item()), (gtBox[2].item(), gtBox[3].item()), (0, 255, 0), 2)
            cv2.imwrite(f"maskCE_{i}.jpg", mask[:,:,::-1])
            cv2.imwrite(f"img_{i}.jpg", img[:,:,::-1])

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        #import pdb
        #pdb.set_trace()
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
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

        # compute contrast loss
        #contrastive_loss = self.objective['contrast'](pred_dict['pred_contrast'])
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        #loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + self.loss_weight['contrast'] * contrastive_loss['loss_contrast'] + self.loss_weight['contrast_aux'] * contrastive_loss['loss_contrast_aux']
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      #"Loss/Contrast": contrastive_loss['loss_contrast'].item(),
                      #"Loss/Contrast_aux" : contrastive_loss['loss_contrast_aux'].item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
