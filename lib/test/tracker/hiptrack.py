import math
from thop import profile
from thop.utils import clever_format
from lib.models.hiptrack import build_hiptrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d, visualizeHanning
from lib.train.data.processing_utils import sample_target
import time
# for debug
import cv2
import os
import numpy as np
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import random

class HIPTrack(BaseTracker):
    def __init__(self, params, dataset_name, visualize_during_infer=False):
        super(HIPTrack, self).__init__(params)
        network = build_hiptrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.network.set_eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.visualize_during_infer = visualize_during_infer
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

    def initialize(self, image, info: dict, videoName: str):
        if self.visualize_during_infer:
            if not os.path.exists("./VisualizeResults"):
                os.makedirs("./VisualizeResults")
            if not os.path.exists(os.path.join("./VisualizeResults", videoName)):
                os.makedirs(os.path.join("./VisualizeResults", videoName))
                self.visualizeRootPath = os.path.join("./VisualizeResults", videoName)
                os.makedirs(os.path.join("./VisualizeResults", videoName, "originalImg"))
                os.makedirs(os.path.join("./VisualizeResults", videoName, "scoreMap"))
                os.makedirs(os.path.join("./VisualizeResults", videoName, "topkBoxes"))
            else:
                self.visualizeRootPath = os.path.join("./VisualizeResults", videoName)
                if not os.path.exists(os.path.join("./VisualizeResults", videoName, "topkBoxes")):
                    os.makedirs(os.path.join("./VisualizeResults", videoName, "topkBoxes"))
        # forward the template once
        self.seqName = videoName
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size) #推理过程中没有Jitter，直接crop区域再resize到output_sz
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr) #Normalize
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
            self.template_bbox_cropped = template_bbox
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes: #一般是False
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None, dynamicUpdateTemplate=False, gt_crop=False):
        #import pdb
        #pdb.set_trace()
        H, W, _ = image.shape
        self.frame_id += 1
        if not gt_crop:
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
        else:
            if info['previous_gt'][-1] == 0 and info['previous_gt'][-2] == 0:
                info['previous_gt'] = self.state
            if info['previous_gt'][-1] < 0 or info['previous_gt'][-2] < 0:
                info['previous_gt'] = self.state
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, list(info['previous_gt']), self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            #if not os.path.exists(f"GT_SearchArea/{self.seqName}"):
            #    os.makedirs(f"GT_SearchArea/{self.seqName}")
            #cv2.imwrite(f"GT_SearchArea/{self.seqName}/{self.frame_id}.jpg", x_patch_arr)
        if self.visualize_during_infer:
            cv2.imwrite(os.path.join(self.visualizeRootPath, "originalImg", f"{self.frame_id}_searchImgOriginal.jpg"), x_patch_arr)
            cv2.imwrite(os.path.join(self.visualizeRootPath, "originalImg", f"{self.frame_id}_seachImgMaskOriginal.jpg"), np.expand_dims(x_amask_arr.astype(np.int8) * 255, 2).repeat(3, axis=2))

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        #cv2.imwrite(f"./dynamic_template/template_{self.frame_id}.jpg", self.z_patch_arr)
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            #import pdb
            #pdb.set_trace()
            #from torchsummary import summary
            #summary(self.network, template=self.z_dict1.tensors, template_boxes=self.template_bbox_cropped, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
#
            out_dict = self.network.forward_track(index=self.frame_id,
                template=self.z_dict1.tensors, template_boxes=self.template_bbox_cropped, search=x_dict.tensors, ce_template_mask=self.box_mask_z, ce_keep_rate=None, searchRegionImg=x_patch_arr, info=self.seqName)
            #from fvcore.nn import FlopCountAnalysis, parameter_count_table
            #import pdb; pdb.set_trace()
            #flops = FlopCountAnalysis(self.network, (self.frame_id, self.z_dict1.tensors, self.template_bbox_cropped, x_dict.tensors, self.box_mask_z, None, x_patch_arr, self.seqName))
            #print(flops.total())
            #T_w = 500
            #T_t = 1000
            #print("testing speed ...")
            #torch.cuda.synchronize()
            #with torch.no_grad():
            #    # overall
            #    for i in range(T_w):
            #        _ = self.network(index=self.frame_id, template=self.z_dict1.tensors, template_boxes=self.template_bbox_cropped, search=x_dict.tensors, ce_template_mask=self.box_mask_z, ce_keep_rate=None, searchRegionImg=x_patch_arr, info=self.seqName)
#
            #    start = time.time()
            #    for i in range(T_t):
            #        _ = self.network(index=self.frame_id, template=self.z_dict1.tensors, template_boxes=self.template_bbox_cropped, search=x_dict.tensors, ce_template_mask=self.box_mask_z, ce_keep_rate=None, searchRegionImg=x_patch_arr, info=self.seqName)
#
            #    torch.cuda.synchronize()
            #    end = time.time()
            #    avg_lat = (end - start) / T_t
            #    print("The average overall latency is %.2f ms" % (avg_lat * 1000))
            #    print("FPS is %.2f fps" % (1. / avg_lat))

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.visualize_during_infer:
            #import pdb
            #pdb.set_trace()
            normed_score_map = np.uint8(255 * pred_score_map[0][0].cpu().detach().numpy())
            normed_score_map = np.stack((normed_score_map, normed_score_map, normed_score_map), axis=2)
            normed_score_map = cv2.resize(normed_score_map, (384, 384))
            normed_score_map_tensor = torch.from_numpy(normed_score_map)
            maxScore = pred_score_map[0][0].max().item()
            maxScoreIndex = torch.nonzero(normed_score_map_tensor == normed_score_map_tensor.max())[0][:2]
            normed_score_map = cv2.applyColorMap(normed_score_map, cv2.COLORMAP_JET)
            img_add_normed_score_map = cv2.addWeighted(x_patch_arr, 0.35, normed_score_map, 0.65, 0)
            cv2.putText(img_add_normed_score_map, f"{maxScore}", (maxScoreIndex[0].item(), maxScoreIndex[1].item()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(self.visualizeRootPath, "scoreMap", f"{self.frame_id}_scoreMapOriginal.jpg"), img_add_normed_score_map)

        response = self.output_window * pred_score_map

        if self.visualize_during_infer:
            normed_response_map = np.uint8(255 * response[0][0].cpu().detach().numpy())
            normed_response_map = np.stack((normed_response_map, normed_response_map, normed_response_map), axis=2)
            normed_response_map = cv2.resize(normed_response_map, (384, 384))
            normed_response_map = cv2.applyColorMap(normed_response_map, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.visualizeRootPath, "scoreMap", f"{self.frame_id}_responseMap.jpg"), normed_response_map)

            img_add_score = cv2.addWeighted(x_patch_arr, 0.4, normed_score_map, 0.6, 0)
            cv2.imwrite(os.path.join(self.visualizeRootPath, "scoreMap", f"{self.frame_id}_vis.jpg"), img_add_score)

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        #cv2.rectangle(x_patch_arr, 
        #                (int(pred_boxes[0][0] * 384 - 0.5 * pred_boxes[0][2] * 384), int(pred_boxes[0][1] * 384 - 0.5 * pred_boxes[0][3] * 384)), 
        #                (int(pred_boxes[0][0] * 384 + 0.5 * pred_boxes[0][2] * 384), int(pred_boxes[0][1] * 384 + 0.5 * pred_boxes[0][3] * 384)), 
        #                (0, 255, 0), 2
        #                )
        #if not os.path.exists(f"./GT_Search_Area_Raw_Result/{self.seqName}"):
        #    os.makedirs(f"./GT_Search_Area_Raw_Result/{self.seqName}")
        #cv2.imwrite(f"./GT_Search_Area_Raw_Result/{self.seqName}/{self.frame_id}.jpg", x_patch_arr)
        #topk_boxes = (out_dict['topk_pred_boxes'].mean(dim=0) * self.params.search_size / resize_factor).tolist()
        
        # get the final box result
        #self.prev_gt = info['previous_gt']
        self.state = clip_box(self.map_box_back(pred_box, resize_factor, gt_crop=gt_crop), H, W, margin=10)
        #print(f"In frame: {self.frame_id}, the box is: {self.state}")

        topk_states = []

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

        if dynamicUpdateTemplate:
            self.runtime_update_template(image=image, info=info)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_forward(self, kalman_pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3] #在原始图像里的坐标
        cx, cy, w, h = kalman_pred_box[0] + 0.5 * kalman_pred_box[2], kalman_pred_box[1] + 0.5 * kalman_pred_box[3], kalman_pred_box[2], kalman_pred_box[3]
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx - (cx_prev - half_side)
        cy_real = cy - (cy_prev - half_side)
        cx_real *= resize_factor
        cy_real *= resize_factor
        w *= resize_factor
        h *= resize_factor
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h] #cxcywh -> tlwh

    def map_box_back(self, pred_box: list, resize_factor: float, gt_crop : bool = False):
        if gt_crop:
            cx_prev, cy_prev = self.prev_gt[0] + 0.5 * self.prev_gt[2], self.prev_gt[1] + 0.5 * self.prev_gt[3]
        else:
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


def get_tracker_class():
    return HIPTrack
