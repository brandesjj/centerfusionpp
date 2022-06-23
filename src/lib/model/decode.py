from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from distutils.log import debug
import imp

import torch
import torch.nn as nn

from .utils import _gather_feat, _tranpose_and_gather_feat
from .utils import _nms, _topk, _topk_channel

# DEBUG#
from utils.post_process import get_alpha
from utils.ddd_utils import alpha2rot_y
from utils.image import get_affine_transform, transform_preds_with_trans

def _update_kps_with_hm(
  kps, output, batch, num_joints, K, bboxes=None, scores=None):
  if 'hm_hp' in output:
    hm_hp = output['hm_hp']
    hm_hp = _nms(hm_hp)
    thresh = 0.2
    kps = kps.view(batch, K, num_joints, 2).permute(
        0, 2, 1, 3).contiguous() # b x J x K x 2
    reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
    if 'hp_offset' in output or 'reg' in output:
        hp_offset = output['hp_offset'] if 'hp_offset' in output \
                    else output['reg']
        hp_offset = _tranpose_and_gather_feat(
            hp_offset, hm_inds.view(batch, -1))
        hp_offset = hp_offset.view(batch, num_joints, K, 2)
        hm_xs = hm_xs + hp_offset[:, :, :, 0]
        hm_ys = hm_ys + hp_offset[:, :, :, 1]
    else:
        hm_xs = hm_xs + 0.5
        hm_ys = hm_ys + 0.5
    
    mask = (hm_score > thresh).float()
    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
        2).expand(batch, num_joints, K, K, 2)
    dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
    min_dist, min_ind = dist.min(dim=3) # b x J x K
    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
        batch, num_joints, K, 1, 2)
    hm_kps = hm_kps.gather(3, min_ind)
    hm_kps = hm_kps.view(batch, num_joints, K, 2)        
    mask = (hm_score < thresh)
    
    if bboxes is not None:
      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
              (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
    else:
      l = kps[:, :, :, 0:1].min(dim=1, keepdim=True)[0]
      t = kps[:, :, :, 1:2].min(dim=1, keepdim=True)[0]
      r = kps[:, :, :, 0:1].max(dim=1, keepdim=True)[0]
      b = kps[:, :, :, 1:2].max(dim=1, keepdim=True)[0]
      margin = 0.25
      l = l - (r - l) * margin
      r = r + (r - l) * margin
      t = t - (b - t) * margin
      b = b + (b - t) * margin
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
              (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
      # sc = (kps[:, :, :, :].max(dim=1, keepdim=True) - kps[:, :, :, :].min(dim=1))
    # mask = mask + (min_dist > 10)
    mask = (mask > 0).float()
    kps_score = (1 - mask) * hm_score + mask * \
      scores.unsqueeze(-1).expand(batch, num_joints, K, 1) # bJK1
    kps_score = scores * kps_score.mean(dim=1).view(batch, K)
    # kps_score[scores < 0.1] = 0
    mask = mask.expand(batch, num_joints, K, 2)
    kps = (1 - mask) * hm_kps + mask * kps
    kps = kps.permute(0, 2, 1, 3).contiguous().view(
        batch, K, num_joints * 2)
    return kps, kps_score
  else:
    return kps, kps



## Decoder with Radar point cloud fusion support
def fusion_decode(output, K=100, opt=None):
  """
  3D box encoder.
  Assemble 2D bounding box and overwrite primary with secondary head.
  """
  if not ('hm' in output):
    return {}

  if opt.zero_tracking:
    output['tracking'] *= 0
  
  heat = output['hm']
  batch, cat, height, width = heat.size()

  heat = _nms(heat) # get peaks (filtered heat map to only keep maxima in every 3x3 region)
  scores, inds, clses, ys0, xs0 = _topk(heat, K=K)  # keep top K peaks

  clses  = clses.view(batch, K)  
  scores = scores.view(batch, K) 
  bboxes = None
  cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2) # x-y-coordinates of top K center points
  ret = {'scores': scores, 'clses': clses.float(), 
         'xs': xs0, 'ys': ys0, 'cts': cts} # store results to return
  if 'reg' in output: # local offset due to downsampling in backbone
    reg = output['reg']
    reg = _tranpose_and_gather_feat(reg, inds) # only use offsets to top K peaks
    reg = reg.view(batch, K, 2) 
    xs = xs0.view(batch, K, 1) + reg[:, :, 0:1] # add predicted offset
    ys = ys0.view(batch, K, 1) + reg[:, :, 1:2] # add predicted offset
  else:
    xs = xs0.view(batch, K, 1) + 0.5
    ys = ys0.view(batch, K, 1) + 0.5
  # No need to store xs and ys in ret since center point is recomputed
  # from bbox. Bbox in turn is computed with offset compensated.

  if 'wh' in output:
    wh = output['wh']
    wh = _tranpose_and_gather_feat(wh, inds) # B x K x (F) | only use predicted 2D bboxes to top K peaks
    
    wh = wh.view(batch, K, 2) 
    wh[wh < 0] = 0 # only allow positive dimensions 
    if wh.size(2) == 2 * cat: # cat spec
      wh = wh.view(batch, K, -1, 2)
      cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
      wh = wh.gather(2, cats.long()).squeeze(2) # B x K x 2
    else:
      pass
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)  # compute 2D bboxes
    ret['bboxes'] = bboxes
 
  if 'ltrb' in output: # ltrb: left top right bottom | additional offset
    ltrb = output['ltrb']
    ltrb = _tranpose_and_gather_feat(ltrb, inds) # B x K x 4
    ltrb = ltrb.view(batch, K, 4)
    bboxes = torch.cat([xs0.view(batch, K, 1) + ltrb[..., 0:1], 
                        ys0.view(batch, K, 1) + ltrb[..., 1:2],
                        xs0.view(batch, K, 1) + ltrb[..., 2:3], 
                        ys0.view(batch, K, 1) + ltrb[..., 3:4]], dim=2)
    ret['bboxes'] = bboxes

  ## Decode depth with depth residual support
  if 'dep' in output:
    dep = output['dep']
    dep = _tranpose_and_gather_feat(dep, inds) # B x K x (C) 
    # dep = dep.view(batch, K, -1)
    # dep[dep < 0] = 0
    cats = clses.view(batch, K, 1, 1)
    if dep.size(2) == cat: # cat spec
      dep = dep.view(batch, K, -1, 1) # B x K x C x 1
      dep = dep.gather(2, cats.long()).squeeze(2) # B x K x 1
    
    # add depth residuals to estimated depth values
    if 'dep_sec' in output:
      dep_sec = output['dep_sec']
      dep_sec = _tranpose_and_gather_feat(dep_sec, inds) # B x K x [C] | only keep depth values to top K peaks
      if dep_sec.size(2) == cat: # cat spec
        dep_sec = dep_sec.view(batch, K, -1, 1) # B x K x C x 1
        dep_sec = dep_sec.gather(2, cats.long()).squeeze(2) # B x K x 1
        dep_sec_mask = torch.tensor(dep_sec_mask, device=dep_sec.device).unsqueeze(0).unsqueeze(0).unsqueeze(3)
      dep = dep_sec # overwrite primary head with secondary head 
    
    ret['dep'] = dep
  

  regression_heads = ['tracking', 'rot', 'dim', 'amodel_offset',
    'nuscenes_att', 'velocity', 'rot_sec']

  for head in regression_heads:
    if head in output:
      ret[head] = _tranpose_and_gather_feat(
        output[head], inds).view(batch, K, -1)
  
  if 'rot_sec' in output:
    ret['rot'] = ret['rot_sec'] # overwrite primary head with secondary head

  if 'ltrb_amodel' in output:
    ltrb_amodel = output['ltrb_amodel']
    ltrb_amodel = _tranpose_and_gather_feat(ltrb_amodel, inds) # B x K x 4
    ltrb_amodel = ltrb_amodel.view(batch, K, 4)
    bboxes_amodel = torch.cat([xs0.view(batch, K, 1) + ltrb_amodel[..., 0:1], 
                          ys0.view(batch, K, 1) + ltrb_amodel[..., 1:2],
                          xs0.view(batch, K, 1) + ltrb_amodel[..., 2:3], 
                          ys0.view(batch, K, 1) + ltrb_amodel[..., 3:4]], dim=2)
    ret['bboxes_amodel'] = bboxes_amodel
    ret['bboxes'] = bboxes_amodel

  if 'hps' in output:
    kps = output['hps']
    num_joints = kps.shape[1] // 2
    kps = _tranpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs0.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys0.view(batch, K, 1).expand(batch, K, num_joints)
    kps, kps_score = _update_kps_with_hm(
      kps, output, batch, num_joints, K, bboxes, scores)
    ret['hps'] = kps
    ret['kps_score'] = kps_score

  if 'pre_inds' in output and output['pre_inds'] is not None:
    pre_inds = output['pre_inds'] # B x pre_K
    pre_K = pre_inds.shape[1]
    pre_ys = (pre_inds / width).int().float()
    pre_xs = (pre_inds % width).int().float()

    ret['pre_cts'] = torch.cat(
      [pre_xs.unsqueeze(2), pre_ys.unsqueeze(2)], dim=2)
  
  return ret
