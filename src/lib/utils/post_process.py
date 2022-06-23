from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from .image import transform_preds_with_trans, get_affine_transform
from .ddd_utils import ddd2locrot, compute_corners_3d
from .ddd_utils import project_to_image, rot_y2alpha
import numba
import math

def get_alpha(rot):
  """
  Revert from sin/cos representation to absolute angle
  tan = sin / cos -> atan gets argument back -> atan2 for robustness
  rot : classification, sin & cos of the relative angle for each bin
  return : absolute angle of object either in bin 1 or bin 2 depending on wether
           the classification of either is higher.
 
  rot : (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
                bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  return (B)
  """
  # The bins refer to themselves in the second classification variable.
  # rot has 6 variables and thus rot[:,1] is
  # giving the probability to be in bin 1. rot[:,0] the probability to be in 
  # bin 2. rot[:,4] the probability to be in bin 1. rot[:,5] the probability
  # to be in bin 2. 
  idx = rot[:, 1] > rot[:, 5] # boolean stating which bin has more certainty in its decision whether the angle is in ITS OWN bin. Would be the same as rot[:,0] > rot[:,4]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)

def points_in_bbox(bbox, pc):
  pass

def generic_post_process(
  opt, dets, c, s, h, w, num_classes, calibs=None, height=-1, width=-1, is_gt=False):
  """
  :param opt: Options (opt.py)
  :param dets: Detections from the network
  :param c: Centerpoint in the original image size (e.g. [800,450] for image size)
  :param s: scale
  :param h: height of the heatmap / image in reduced size
  :param w: width of heatmap / image in reduced size
  :param calibs: camera matrix P [3x4]
  """
  if not ('scores' in dets):
    # No objects 
    return [{}], [{}]
  ret = []

  # Loop over samples in batch
  for i in range(len(dets['scores'])):  
    preds = []
    # Get inverse transformation (output size -> input size)
    trans = get_affine_transform(
      c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
    # Loop over top K predictions / annotations
    for j in range(len(dets['scores'][i])):   
      if dets['scores'][i][j] < opt.out_thresh:
        break
      item = {}
      item['score'] = dets['scores'][i][j]
      item['class'] = int(dets['clses'][i][j]) + 1
      # Transform to original image size
      item['ct'] = transform_preds_with_trans(
        dets['cts'][i][j].reshape(1, 2), trans).reshape(2)

      if 'tracking' in dets:
        # Transform to original image size
        tracking = transform_preds_with_trans(
          (dets['tracking'][i][j] + dets['cts'][i][j]).reshape(1, 2), 
          trans).reshape(2)
        item['tracking'] = tracking - item['ct']

      if 'bboxes' in dets:
        # Transform to original image size
        bbox = transform_preds_with_trans(
          dets['bboxes'][i][j].reshape(2, 2), trans).reshape(4)
        item['bbox'] = bbox

      if 'hps' in dets:
        # Transform to original image size
        pts = transform_preds_with_trans(
          dets['hps'][i][j].reshape(-1, 2), trans).reshape(-1)
        item['hps'] = pts

      if 'dep' in dets and len(dets['dep'][i]) > j:
        item['dep'] = dets['dep'][i][j]
        if len(item['dep'])>1:
          item['dep'] = item['dep'][0]
      
      if 'dim' in dets and len(dets['dim'][i]) > j:
        item['dim'] = dets['dim'][i][j]

      if 'rot' in dets and len(dets['rot'][i]) > j:
        item['alpha'] = get_alpha(dets['rot'][i][j:j+1])[0]
      
      if 'rot' in dets and 'dep' in dets and 'dim' in dets \
        and len(dets['dep'][i]) > j:
        if 'amodel_offset' in dets and len(dets['amodel_offset'][i]) > j:
          #  Apply predicted amodel offset to shift center point 
          # If object is outside of frame the center point has to be shifted
          # from the predicted center point 
          ct_output = dets['bboxes'][i][j].reshape(2, 2).mean(axis=0)
          amodel_ct_output = ct_output + dets['amodel_offset'][i][j]
          # Transform to original image size
          ct = transform_preds_with_trans(
            amodel_ct_output.reshape(1, 2), trans).reshape(2).tolist()
        else:
          # If there is no prediction for amodel offset just use bbox
          bbox = item['bbox']
          ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        # Store center point
        item['ct'] = ct
        # Get 3D location and rotation around y-aixs of camera CS
        item['loc'], item['rot_y'] = ddd2locrot(
          ct, item['alpha'], item['dim'], item['dep'], calibs[i])
      
      preds.append(item)

    if 'nuscenes_att' in dets:
      for j in range(len(preds)): # basically also loop over predictions / annotations
        preds[j]['nuscenes_att'] = dets['nuscenes_att'][i][j]

    if 'velocity' in dets:
      for j in range(len(preds)): # basically also loop over predictions / annotations
        vel = dets['velocity'][i][j]
        if opt.pointcloud and not is_gt: 
          ## put velocity in the same direction as box orientation (but not for ground truth)
          V = math.sqrt(vel[0]**2 + vel[2]**2) # absolute velocity 
          vel[0] = np.cos(preds[j]['rot_y']) * V
          vel[2] = -np.sin(preds[j]['rot_y']) * V
        preds[j]['velocity'] = vel[:3] 
    
    ret.append(preds)
  
  return ret