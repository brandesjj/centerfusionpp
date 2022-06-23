# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat, _nms, _topk
import torch.nn.functional as F
from utils.image import draw_umich_gaussian

# NOTE: M = max_num_objs

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _neg_loss(pred, gt):
  ''' Reimplemented focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _only_neg_loss(pred, gt):
  gt = torch.pow(1 - gt, 4)
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
  return neg_loss.sum()

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()
    self.only_neg_loss = _only_neg_loss

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    neg_loss = self.only_neg_loss(out, target)
    pos_pred_pix = _tranpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      # Also punish predictions even if there is no annotation since 
      # center points are used to filter all other parameters.
      # This means when no center point is predicted all other 
      # parameters get surpressed as well. Thus the other heads don't
      # need to train to NOT classify something.
      return - neg_loss 

    return - (pos_loss + neg_loss) / num_pos


def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
  regr_loss = regr_loss / (num + 1e-6)
  return regr_loss


class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-6)
    return loss


class WeightedBCELoss(nn.Module):
  def __init__(self):
    super(WeightedBCELoss, self).__init__()
    self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

  def forward(self, output, mask, ind, target):
    # output: B x F x H x W
    # ind: B x M
    # mask: B x M x F
    # target: B x M x F
    pred = _tranpose_and_gather_feat(output, ind) # B x M x F
    loss = mask * self.bceloss(pred, target)
    loss = loss.sum() / (mask.sum() + 1e-6)
    return loss


class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres, opt):
    """ 
    Forward function of rotation loss. Defined in Mousavian et. al thorugh bins that
    represent the searched angle with n*4 parameters. Here we choose n=2 bins.
    'output' is the output of the rot head (prim and sec) which have a img in and out size (H&W) of 
    200x112 by default. The rot needs 8 parameters (=C) for regression.
    'mask' surpresses predictions for annotations that don't have any ground truth for rotation
    labeled or just don't exist in this sample. Each sample can have up to 128 annotations. 
    If there exist less, 'mask' makes sure that the network doesn't learn with the extra annotations.
    'ind' : 
    'rotbin' : ground truth for bin classification
    'rotres' : ground truth for relative angle (absolute angle relative to center of bin)

    output : [B, C, W, H] 
    mask : [B, 128]
    ind : [B, 128]
    rotbin : [B, 128, 2]
    rotbin : [B, 128, 2]
    """
    # Get predictions for rotation for 128 annotations with 8 dimension per prediction
    pred = _tranpose_and_gather_feat(output, ind) # shape [B, 128, 8]
    loss = compute_rot_loss(pred, rotbin, rotres, mask, opt)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean')

def compute_bin_loss(output, target, mask, opt):
    """
    Compute loss from classifying if angle is in this or in the other bin with cross entropy.
    Use the prediction if it's in this bin AND if it's in the other bin because bins overlap
    and the angle can be in both bins.
    Mask predictions by wether or not the annotations even have an angle labeled or not.
    Don't learn from making a prediction when there is no target.
    """
    if opt.custom_rotbin_loss:
      # loss
      nonzero_idx = mask.nonzero()[:,0].long()
      if nonzero_idx.shape[0] > 0: # if there are any annotations with a labeled angle
        output_mod = output.index_select(0, nonzero_idx)
        target_mod = target.index_select(0, nonzero_idx)
        loss_mod = F.cross_entropy(output_mod, target_mod, reduction='mean') 
      else: # loss would be nan if computed normally when no annotation is given 
        loss_mod = torch.tensor(0.0).cuda() # set to different grad_fn but not relevant since loss is zero
      # print("Modified Loss: ", loss_mod) 
    else:
      mask_original = mask.expand_as(output)
      output_original = output * mask_original.float()
      loss_original = F.cross_entropy(output_original, target, reduction='mean') 
      # print("Original Loss: ", loss_original)
    return loss_mod
    

def compute_rot_loss(output, target_bin, target_res, mask, opt):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    
    # Squeeze batches together (first two dimension of args)
    output = output.view(-1, output.shape[2]) # shape [Bx128, 8]
    target_bin = target_bin.view(-1, target_bin.shape[2]) # shape [Bx128, 2]
    target_res = target_res.view(-1, target_res.shape[2]) # shape [Bx128, 2]
    mask = mask.view(-1, 1) # shape [Bx128, 1] | nonzero for every annotation with a labeled angle

    ## Compute loss from classification
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask, opt)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask, opt)

    ## Compute loss from angle (Delta theta)
    loss_res = torch.zeros_like(loss_bin1)
    # First bin
    targets_in_bin_1 = target_bin[:, 0].nonzero() # get all targets for which there is an angle in the first bin
    if targets_in_bin_1.shape[0] > 0: # if there is at least one ground truth angle in the first bin 
        # only compute loss for pred where there is a target, i.e. mask it
        idx1 = targets_in_bin_1[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long()) # just take the sin/cos pred of the angle IF the gt angle is in this bin 
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        # Compute sin & cos of relative angle as in Mousavian et. al and take the loss
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    # Second bin
    targets_in_bin_2 = target_bin[:, 1].nonzero() # get all targets for which there is an angle in the second bin
    if targets_in_bin_2.shape[0] > 0: # if there is at least one ground truth angle in the second bin
        # only compute loss for pred where there is a target, i.e. mask it
        idx2 = targets_in_bin_2[:, 0] 
        valid_output2 = torch.index_select(output, 0, idx2.long()) # just take the sin/cos pred of the angle IF the network predicted it to be in this bin 
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        # Compute sin & cos of relative angle as in Mousavian et. al and take the loss
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
  

## Loss function for depth with support for class-based depth
class DepthLoss(nn.Module):
  def __init__(self, opt=None):
    super(DepthLoss, self).__init__()

  def forward(self, output, target, ind, mask, cat):
    '''
    Arguments:
      out: B x C x H x W
      target: B x M x C
      mask: B x M x 1
      ind: B x M
      cat (category id for peaks): B x M
    '''
    pred = _tranpose_and_gather_feat(output, ind) # B x M x (C)
    if pred.shape[2] > 1: # if depth is predicted for multiple classes
      pred = pred.gather(2, cat.unsqueeze(2)) # B x M
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-6)
    return loss


class LFALoss(nn.Module):

  def __init__(self):
    super(LFALoss, self).__init__()

  def forward(self, output, target, snap_mask=None, mask=None, weights=None, 
              lfa_with_ann=True):
    '''
    C = len(opt.pc_feat_lvl)
    M = max_objs for this dataset
    Arguments:
      out: tensor in phase='train': B x M x C   [pc_lfa_feat]
                  in phase='val': B x C x W x H [pc_box_hm]
      target: tensor in phase='train': B x M x C   [pc_lfa_feat]
                     in phase='val': B x C x W x H [pc_box_hm]
      snap_mask: tensor B x M Mask out objects where there is no point in frustum. 
                 Don't learn from them.
      snap: tensor B x M Mask out placeholder objects. Don't learn from them.
      weight: tensor C To weight velocity error more
      :param phase: String Phase is either 'train' or 'val'. For 'val' 
                    calculate loss with pc BB hm but in 'train' only if grad 
                    with predicted pc features. This is because in validation
                    we cannot compare prediction with target. The tensors are 
                    filled with values corresponding to possibly different
                    objects. The output and target can be compared in training
                    because there the pc features are predicted for the objects
                    sorted in the same order as the targets are.
    '''
    if lfa_with_ann:
      # Get all valid objects by filtering with both masks
      combined_mask = snap_mask * mask
      output *= combined_mask * weights # weight vel more
      target *= combined_mask * weights # weight vel more
      # dep_loss = F.l1_loss(output[:,:,0], target[:,:,0], reduction='sum')
      # vx_loss = F.l1_loss(output[:,:,1], target[:,:,1], reduction='sum')
      # vz_loss = F.l1_loss(output[:,:,2], target[:,:,2], reduction='sum')
      # print('')
      # print('dep loss: ', dep_loss)
      # print('vx loss: ', vx_loss)
      # print('vz loss: ', vz_loss)
      loss = F.l1_loss(output, target, reduction='sum')
      # print(f'rel (dep|vx|vz): {dep_loss/loss} | {vx_loss/loss} | {vz_loss/loss}')
      # Only mean over valid objects
      loss /= combined_mask.sum() + 1e-6
    else:
      loss = F.l1_loss(output, target, reduction='sum')
    
    if torch.isnan(loss):
      return torch.tensor(0.0).cuda()
    else:
      return loss
    

