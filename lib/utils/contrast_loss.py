from abc import ABC
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, contrastItems):
        contras_loss = 0
        aux_loss = 0
        if len(contrastItems) == 0:
            return {
                'loss_contrast' : torch.tensor(0.0).cuda(),
                'loss_contrast_aux' : torch.tensor(0.0).cuda() 
            }
        for contrastItem in contrastItems:
            if len(contrastItem['contrast']) == 0:
                continue
            pred = contrastItem['contrast'].permute(1, 0)
            label = contrastItem['label'].unsqueeze(0)
            pos_inds = (label == 1)
            neg_inds = (label == 0)
            pred_pos = pred * pos_inds.float()
            pred_neg = pred * neg_inds.float()
            # use -inf to mask out unwanted elements.
            pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
            pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

            _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
            _neg_expand = pred_neg.repeat(1, pred.shape[1])
            # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
            x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0) 
            contras_loss += torch.logsumexp(x, dim=1)

            aux_pred = contrastItem['aux_consin'].permute(1,0)
            aux_label = contrastItem['aux_label'].unsqueeze(0)
            aux_loss += (torch.abs(aux_pred - aux_label)**2).mean()
        #print("aux_loss is : ", aux_loss)
        #import pdb
        #pdb.set_trace()
        losses = {
            'loss_contrast' : contras_loss.sum() / len(contrastItems),
            'loss_contrast_aux' : aux_loss / len(contrastItems) 
        }
        return losses