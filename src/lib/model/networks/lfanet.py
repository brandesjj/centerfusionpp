import copy
import imp
from math import log
import numpy as np
from pyparsing import rest_of_line
import torch
from torch import nn
from torch.nn import Conv2d, Sequential, ReLU, MaxPool2d, BatchNorm2d, Linear
import os
import matplotlib.pyplot as plt

from model.utils import _nms, _sigmoid, _topk, _tranpose_and_gather_feat
from utils.ddd_utils import ddd2locrot, ddd2locrot_torch, v_to_vrad_torch
from utils.image import get_affine_transform, transform_preds_with_trans_torch, transform_preds_with_trans_torch_minibatch
from utils.pointcloud import get_alpha, get_dist_thresh, get_dist_thresh_torch
from utils.eval_frustum import EvalFrustum, debug_lfa_frustum
from utils.snapshot import generate_snap_BEV_torch, generate_snap_proj_torch

from model.networks.pointnetpp import PointNetPP, get_loss

import cv2

try:
    from .DCNv2.dcn_v2 import DCN
except:
    print('Import DCN in dla.py failed')
    print(f'Current location is {os.getcwd()}')
    DCN = None

class Flatten(nn.Module):
  """
  Helper function to include Flatten() Module in Sequential()
  """
  def forward(self, input):
    return input.view(input.size(0), -1)

class CustomGlobalAvgPool(nn.Module):
  """
  Apparantely faster than torch.nn.AvgPool2d
  Source: https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721
  """
  def forward(self, x):
    return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

# import time
class LFANet(nn.Module):
  def __init__(self, opt):
    self.opt = opt
    self.num_stacks = opt.num_stacks
    self.res_in = opt.snap_resolution # square input resolution

    self.pc_nr_feat = opt.lfa_pc_nr_feat
    nr_channels = opt.num_lfa_filters
    nr_channels_in = opt.lfa_channel_in
    super(LFANet, self).__init__()

    # Define Network 
    # Number of layers and scaling is fixed for simplicity
    # since it has to start and end up in specific resolutions
    
    # Input has to size a power of 2. Check with bit trick 
    # (see https://stackoverflow.com/questions/600293/)
    assert (self.res_in != 0) and ((self.res_in & (self.res_in - 1)) == 0), \
      "Wrong input dimension. It has to be a power of 2."
    
    if opt.use_pointnet:
      self.convs = PointNetPP(opt.lfa_channel_in, opt.lfa_pc_nr_feat, normal_channel=False)

    else:
      self.convs = []

      # For dense variantes
      nr_extra_layers = 4 
      # Increase nr of channels proportionally to downscaling of input
      increase_channels = opt.increase_nr_channels
      # Select network type to reduce the snapshot to a single pixel with nr_filter in channels
      if self.opt.lfa_network in ['img','img_dense']:
        kernel_size = 3

        if self.res_in == 2:
          if not opt.not_use_dcn:
            self.convs.append(DCN(nr_channels_in, nr_channels, kernel_size=(2,2),
                                  stride=1, padding=0)) # bias = True by default
          else:
            self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=2,
                                    stride=1, padding=0, bias=True))
          # Batchnorm
          if opt.bn_in_head_arch:
            self.convs.append(BatchNorm2d(nr_channels))
          # ReLU activation layer
          self.convs.append(ReLU())
        else: 
          res_curr = self.res_in # current resolution of the "image"
          for pow in range(1, int(log(self.res_in, 2))):
            # Convolutinal layer with constant amount of channels (keep the same "image" size)
            if not opt.not_use_dcn:
              self.convs.append(DCN(nr_channels_in, nr_channels, kernel_size=(kernel_size,kernel_size),
                                    stride=1, padding=kernel_size // 2)) # bias = True by default
            else:
              self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=True))
            # Batchnorm
            if opt.bn_in_head_arch:
              self.convs.append(BatchNorm2d(nr_channels))
            # ReLU activation layer
            self.convs.append(ReLU())

            # Dense version
            if self.opt.lfa_network == 'img_dense':
              for _ in range(nr_extra_layers):
                if not opt.not_use_dcn:
                  self.convs.append(DCN(nr_channels, nr_channels, kernel_size=(kernel_size,kernel_size),
                                        stride=1, padding=kernel_size // 2)) # bias = True by default
                else:
                  self.convs.append(Conv2d(nr_channels, nr_channels, kernel_size=kernel_size,
                                            stride=1, padding=kernel_size // 2, bias=True))
                # Batchnorm
                if opt.bn_in_head_arch:
                  self.convs.append(BatchNorm2d(nr_channels))
                # ReLU activation layer
                self.convs.append(ReLU())
            
            # Max Pooling layer to reduce "image" size
            reduce_factor = 2**(pow)
            self.convs.append(MaxPool2d(int(min(reduce_factor, res_curr))))

            # If reduced enough (to 1 pixel)
            if reduce_factor >= res_curr:
              break
            # Update "image" size
            res_curr /= reduce_factor
            # Update nr_channels for next layer (increase proportionally)
            nr_channels_in = nr_channels
            if increase_channels:
              nr_channels *= reduce_factor
            nr_extra_layers = nr_extra_layers // 2

      elif self.opt.lfa_network in ['img_global','img_global_avg']:
        kernel_size = 3
        # nr_channels is a list containing the number of layers 
        # and the filters per layer
        layers = nr_channels
        if type(layers) == int:
          layers = [layers]
        for layer, layer_channels in enumerate(layers):
          if layer == 0:
            if not opt.not_use_dcn:
              self.convs.append(DCN(nr_channels_in, layer_channels, kernel_size=(kernel_size,kernel_size),
                                    stride=1, padding=kernel_size // 2)) # bias = True by default
            else:
              self.convs.append(Conv2d(nr_channels_in, layer_channels, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=True))
          else:
            if not opt.not_use_dcn:
              self.convs.append(DCN(layers[layer-1], layer_channels, kernel_size=(kernel_size,kernel_size),
                                    stride=1, padding=kernel_size // 2)) # bias = True by default
            else:
              self.convs.append(Conv2d(layers[layer-1], layer_channels, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=True))
          # Batchnorm
          if opt.bn_in_head_arch:
            self.convs.append(BatchNorm2d(layer_channels))
          # ReLU activation layer
          self.convs.append(ReLU())
        # Use conv layer to map to out channels before pooling!
        self.convs.append(Conv2d(layer_channels, self.pc_nr_feat, kernel_size=1, 
                                stride=1, padding=0, bias=True))

        if opt.lfa_network == 'img_global':
          # Global pooling layer
          self.convs.append(MaxPool2d(self.res_in)) # [B x pc_nr_feat x 1 x 1]
        else:
          # Global average pooling layer
          self.convs.append(CustomGlobalAvgPool()) # [B x pc_nr_feat]


      elif self.opt.lfa_network == 'pc':
        # 7x7 to reduce size by 4 and then 1 (32x32 -> 7x7)
        if not opt.not_use_dcn:
          self.convs.append(DCN(nr_channels_in, nr_channels, kernel_size=(7,7),
                                stride=4, padding=1)) # bias = True by default
        else:
          self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=7,
                                  stride=4, padding=1, bias=True))
        # Batchnorm
        if opt.bn_in_head_arch:
          self.convs.append(BatchNorm2d(nr_channels))
        # ReLU activation layer
        self.convs.append(ReLU())

        # Update nr of channels
        nr_channels_in = nr_channels
        if increase_channels:
          nr_channels *= 4 # close to proportional increase to ratio of downsizing

        # 7x7 to reduce size by 7 (7x7 -> 1x1)
        if not opt.not_use_dcn:
          self.convs.append(DCN(nr_channels_in, nr_channels, kernel_size=(7,7),
                                stride=1, padding=0)) # bias = True by default
        else:
          self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=7,
                                  stride=1, padding=0, bias=True))
        # Batchnorm
        if opt.bn_in_head_arch:
          self.convs.append(BatchNorm2d(nr_channels))
        # ReLU activation layer
        self.convs.append(ReLU())

      elif self.opt.lfa_network == 'pc_dense':
        kernel_size = 7
        # Add more layers
        for _ in range(nr_extra_layers):
          if not opt.not_use_dcn:
            self.convs.append(DCN(nr_channels_in, nr_channels, kernel_size=(kernel_size,kernel_size),
                                  stride=1, padding=kernel_size // 2)) # bias = True by default
          else:
            self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=kernel_size,
                                    stride=1, padding=kernel_size // 2, bias=True))
          # Batchnorm
          if opt.bn_in_head_arch:
            self.convs.append(BatchNorm2d(nr_channels))
          # ReLU activation layer
          self.convs.append(ReLU())
          # Update nr of channels
          nr_channels_in = nr_channels

        if increase_channels:
          # Update nr of channels
          nr_channels *= 4 # close to proportional increase to ratio of downsizing

        # 7x7 to reduce size by 4 and then 1 (32x32 -> 7x7)
        if not opt.not_use_dcn:
          self.convs.append(DCN(nr_channels_in, nr_channels, kernel_size=(kernel_size,kernel_size),
                                stride=4, padding=1)) # bias = True by default
        else:
          self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=7,
                                  stride=4, padding=1, bias=True))
        # Batchnorm
        if opt.bn_in_head_arch:
          self.convs.append(BatchNorm2d(nr_channels))
        # ReLU activation layer
        self.convs.append(ReLU())
        
        # Add more layers
        nr_channels_in = nr_channels # update nr of channels
        nr_extra_layers = nr_extra_layers // 2
        for _ in range(nr_extra_layers):
          if not opt.not_use_dcn:
            self.convs.append(DCN(nr_channels_in, nr_channels, kernel_size=(kernel_size,kernel_size),
                                  stride=1, padding=kernel_size // 2)) # bias = True by default
          else:
            self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=kernel_size,
                                    stride=1, padding=kernel_size // 2, bias=True))
          # Batchnorm
          if opt.bn_in_head_arch:
            self.convs.append(BatchNorm2d(nr_channels))
          # ReLU activation layer
          self.convs.append(ReLU())

        if increase_channels:
          # Update nr of channels
          nr_channels *= 4 # close to proportional increase to ratio of downsizing
        
        # 7x7 to reduce size by 7 (7x7 -> 1x1)
        if not opt.not_use_dcn:
          self.convs.append(DCN(nr_channels_in, nr_channels, kernel_size=(kernel_size,kernel_size),
                                stride=1, padding=0)) # bias = True by default
        else:
          self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=7,
                                  stride=1, padding=0, bias=True))
        # Batchnorm
        if opt.bn_in_head_arch:
          self.convs.append(BatchNorm2d(nr_channels))
        # ReLU activation layer
        self.convs.append(ReLU())
      
      elif self.opt.lfa_network == 'pc_dilated':
        if not opt.not_use_dcn:
          print('WARNING: Network type pc_dilated is not compatible with DCN layers! DCN layers are not used.')
        # 7x7 with dilation (only to 8x8 to allow for higher padding) (32x32 -> 8x8)
        self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=7,
                                stride=3, padding=2, dilation=2, bias=True))
        # Batchnorm
        if opt.bn_in_head_arch:
          self.convs.append(BatchNorm2d(nr_channels))
        # ReLU activation layer
        self.convs.append(ReLU())

        # Update nr of channels
        nr_channels_in = nr_channels
        if increase_channels:
          nr_channels *= 4 # close to proportional increase to ratio of downsizing

        # 8x8 to reduce size by 8 (8x8 -> 1x1)
        self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=8,
                                stride=1, padding=0, bias=True))
        # Batchnorm
        if opt.bn_in_head_arch:
          self.convs.append(BatchNorm2d(nr_channels))
        # ReLU activation layer
        self.convs.append(ReLU())

      elif self.opt.lfa_network == 'pc_dilated_dense':
        if not opt.not_use_dcn:
          print('WARNING: Network type pc_dilated is not compatible with DCN layers! DCN layers are not used.')

        kernel_size = 7

        # Add more layers
        for _ in range(nr_extra_layers):
          self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=kernel_size,
                                  stride=1, padding=6, dilation=2, bias=True))
          # Batchnorm
          if opt.bn_in_head_arch:
            self.convs.append(BatchNorm2d(nr_channels))
          # ReLU activation layer
          self.convs.append(ReLU())
          # Update nr of channels
          nr_channels_in = nr_channels

        if increase_channels:
          # Update nr of channels
          nr_channels *= 4 # proportional increase to ratio of downsizing

        # 7x7 with dilation (only to 8x8 to allow for higher padding) (32x32 -> 8x8)
        self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=7,
                                stride=3, padding=2, dilation=2, bias=True))
        # Batchnorm
        if opt.bn_in_head_arch:
          self.convs.append(BatchNorm2d(nr_channels))
        # ReLU activation layer
        self.convs.append(ReLU())

        # Add more layers
        kernel_size = 7
        nr_channels_in = nr_channels # Update nr of channels
        nr_extra_layers = nr_extra_layers // 2
        for _ in range(nr_extra_layers):
          self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=kernel_size,
                                  stride=1, padding=kernel_size // 2, bias=True))
          # Batchnorm
          if opt.bn_in_head_arch:
            self.convs.append(BatchNorm2d(nr_channels))
          # ReLU activation layer
          self.convs.append(ReLU())

        if increase_channels:
          # Update nr of channels
          nr_channels *= 4 # proportional increase to ratio of downsizing

        # 8x8 to reduce size by 8 (8x8 -> 1x1)
        self.convs.append(Conv2d(nr_channels_in, nr_channels, kernel_size=8,
                                stride=1, padding=0, bias=True))
        # Batchnorm
        if opt.bn_in_head_arch:
          self.convs.append(BatchNorm2d(nr_channels))
        # ReLU activation layer
        self.convs.append(ReLU())

      else:
        raise ValueError("Not supported network type!")

      if not opt.lfa_network in ['img_global','img_global_avg']:
        # Convert channels to nr of pointcloud features w/ fc or conv layer
        if opt.num_lfa_fc_layers > 0:
          # Use FC layers
          self.convs.append(Flatten())
          for _ in range(opt.num_lfa_fc_layers - 1):
            self.convs.append(Linear(nr_channels, nr_channels))
            # Batchnorm
            if opt.bn_in_head_arch:
              self.convs.append(BatchNorm2d(nr_channels))
            # ReLU activation layer
            self.convs.append(ReLU())
          self.convs.append(Linear(nr_channels, self.pc_nr_feat))
        else:
          # Use conv layer
          self.convs.append(Conv2d(nr_channels, self.pc_nr_feat, kernel_size=1, 
                                  stride=1, padding=0, bias=True))


      # Turn into Sequential object to run all at once
      self.convs = Sequential(*self.convs)
      
    print("LFANet structure: ", self.convs)

  def _inference_lfa(self, pred):
    """
    Compute input to LFANet (all precitdions for objects and their properties) by inference.
    Do inference step to assemble amodal center point, 2D BBs size, 3D BBs dimensions, predicted rotation in bin format (!) and class.
    Rotation needs to be converted to local orientation alpha for every object.
    """
    # Do not train LFANet only mode
    ### Init ###
    K = self.opt.max_objs # max number of output objects
    hm = pred['hm'].clone()
    hm = _sigmoid(hm) # convert scores to probabilities
    hm = _nms(hm) # get rid of overlapping predictions by NMS
    batch_size, cat, _, _ = hm.size()
    ### Get top detections by applying inference ###
    # 'top' means the K predictions with the highest certainty for being an
    # object and their respective attributes 
    ## Get top center points from heatmap (called peaks)
    # Center point is predicted in output image size
    # Get inds of those center points in the image-like output as well
    # These inds are used to get the other predictions 
    scores, inds, clses, ys0, xs0 = _topk(hm, K=K)
    xs0 = xs0.view(batch_size, K, 1)
    ys0 = ys0.view(batch_size, K, 1)
    clses = clses.view(batch_size, K, 1, 1)
    scores = scores.view(batch_size, K, 1, 1)
    ## Get top local offset 
    # Predict an offset in position due to downscaling of the image in 
    # the (backbone) network
    if 'reg' in pred:
      local_offset = _tranpose_and_gather_feat(pred['reg'], 
                                              inds).view(batch_size, K, -1)
    else:
      # When there is no local offset set it to zero
      local_offset = torch.zeros((xs0.shape[0], xs0.shape[1], 2),\
                                  device=xs0.device)
    ## Get top amodel offset
    # If object is outside of frame the center point has to be shifted
    # from the predicted center point 
    if 'amodel_offset' in pred:
      amodel_offset = _tranpose_and_gather_feat(pred['amodel_offset'], 
                                                inds).view(batch_size, K, -1)
    else:
      # When there is no amodel offset set it to zero
      amodel_offset = torch.zeros((xs0.shape[0], xs0.shape[1], 2),\
                                  device=xs0.device)
    # Assemble center points
    xs = xs0 + local_offset[:, :, 0:1]
    ys = ys0 + local_offset[:, :, 1:2]
    amodel_xs = xs + amodel_offset[:, :, 0:1]
    amodel_ys = ys + amodel_offset[:, :, 1:2]
    amodel_cts = torch.cat([amodel_xs, amodel_ys], dim=2)
    ## Get top bounding boxes
    wh = _tranpose_and_gather_feat(pred['wh'], inds).view(batch_size, K, 2) # B x K x 2 x (C)
    wh[wh < 0] = 0
    if wh.size(2) == 2 * cat: # handle multiple classes for bounding boxes
      wh = wh.view(batch_size, K, -1, 2)
      cats = clses.expand(batch_size, K, 1, 2)
      wh = wh.gather(2, cats.long()).squeeze(2) # B x K x 2 | only get the top K wh values over all classes and not per class
    # Assemble boxes in downsampled image
    bboxes_amodel = torch.cat([amodel_xs - wh[..., 0:1] / 2, 
                               amodel_ys - wh[..., 1:2] / 2,
                               amodel_xs + wh[..., 0:1] / 2, 
                               amodel_ys + wh[..., 1:2] / 2], dim=2)  # B x K x 4
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)  # B x K x 4
    ## Get top depths 
    out_deps = 1. / (pred['dep'].sigmoid() + 1e-6) - 1.  # transform depths with Eigen et. al (essentially e^(-output['dep'])) since network is trained on transformed depth domain
    deps = _tranpose_and_gather_feat(out_deps, inds) # B x K x (C)
    if deps.size(2) == cat: # handle mutliple classes for depth
      cats = clses
      deps = deps.view(batch_size, K, -1, 1) # B x K x C x 1
      deps = deps.gather(2, cats.long()).squeeze(2) # B x K x 1 | only get the top K depth values over all classes and not per class
    ## Get top dimensions
    dims = _tranpose_and_gather_feat(pred['dim'], inds).view(batch_size, K, -1)
    ## Get top rotations
    rots = _tranpose_and_gather_feat(pred['rot'], inds).view(batch_size, K, -1)

    return amodel_cts, bboxes, bboxes_amodel, deps, dims, rots, clses, scores

  def generate_lfa_pc_box_hm_torch(self, batch, output, phase):
    """
    Forward pass method for LFANet in Validation
    
    :param object: dict Is a detection s
    'dets are either gt or output of Network in form of annotations
    If batch is not None we are in training and use the gt box to compute snap
    Computes pointcloud features from Snapshot of frustum. 

    :param snap: torch tensor Snapshot of frustum with square "image" size. Shape is
                 [B, C, W, H] where B is the batch size, C is the features given from 
                 pointcloud, W and H are the "image" dimensions and should be square
                 since the snapshots are scaled to the same size. To prevent different 
                 scaling of either dimension both should be equal.
    :param output: torch tensor Output of primary heads.
    :param opt: dict Options declared in bash script.
    :param phase: String either 'train' or 'val'. Most importantly for frustum expansion ratios
    
    :return: tensor pc_box_hm Pointcloud heatmap that is the size of the "image" outputted
             from the backbone with additional channels of radar points. The heatmap is
             zero except at 2D bounding box locations where it is filled with the value
             of the radar features predicted from the LFANet.
    :return: tensor lfa_snap_mask Mask out objects (or their frustums) for which the snapshot 
             doesn't contain any radar points. The network should not learn from those 
             snapshots. The mask is used in the loss function to filter them out.
    """
    # ------------------------------------------------ #
    # Init 
    # ------------------------------------------------ #
    device = batch['pc_hm'].device
    pc_box_hm = torch.zeros_like(batch['pc_hm']) # If no radar point is inside frustum pc_box_hm is zero everywhere
    batch_size = pc_box_hm.shape[0]
    if phase == 'val':
      assert batch_size == 1 + int(self.opt.flip_test), "Batch size should be 1 while validating and 2 with flip_test"

    # ------------------------------------------------ #
    # Interference step and forward confident pred to LFANet. 
    # Map its output to pc_box_hm which is forwarded to the sec heads.
    # ------------------------------------------------ #
    amodel_cts, bboxes, bboxes_amodel, deps, dims, rots, clses, scores = self._inference_lfa(output)
    ### Loop over images ###
    # Compute pointcloud heatmap with features per object and bbox per object
    # Iterate over images in batch (img_ind should always be 0)
    for img_ind, [amodel_ct_img, bboxes_img, bboxes_amodel_img, depth_img, dim_img, rot_img, cls_img, score_img] in \
            enumerate(zip(amodel_cts, bboxes, bboxes_amodel, deps, dims, rots, clses, scores)):
      ## Get transformation and calibration stored in batch
      trans = batch['trans_original'][img_ind].clone()
      calib = batch['calib'][img_ind].clone()
      
      ## Convert predicted rotation in bin representation with sin & cos to an angle
      alpha_img = get_alpha(rot_img)
      ## Transform center point to input image size
      ct_img = transform_preds_with_trans_torch_minibatch(amodel_ct_img, trans)
      ## Calculate 3D positions and global orientations of all objects in the image
      loc_img, rot_y_img = ddd2locrot_torch(ct_img, alpha_img, dim_img, depth_img, calib)
      rot_y_img = rot_y_img.unsqueeze(1)
      ## Sort characteristics by depth such that farthest is first. Therefore closer objects don't get overwritten when projected
      if self.opt.sort_det_by_depth:
        idx = torch.argsort(depth_img[:,0], descending=True)
        loc_img = loc_img[idx,:]
        bboxes_img = bboxes_img[idx,:]
        bboxes_amodel_img = bboxes_amodel_img[idx,:]
        dim_img = dim_img[idx,:]
        rot_y_img = rot_y_img[idx,:]
        cls_img = cls_img[idx,:]
        score_img = score_img[idx,:]
      
      ## Get point cloud
      pc_3d = batch['pc_3d'][img_ind][:,0:batch['pc_N'][img_ind]].clone()
      ## Get projected snap
      if self.opt.snap_method == 'proj':
        pc_snap_proj = batch['pc_snap_proj'][img_ind].clone()

      ## Iterate over top K detections (predictions of all parameters) over all classes in image
      for obj_ind, [loc, bbox, bbox_amodel, dim, rot_y, cls, score] in \
        enumerate(zip(loc_img, bboxes_img, bboxes_amodel_img, dim_img, rot_y_img, cls_img, score_img)):
        # If prediction is not accurate enough skip it to save runtime
        if score >= self.opt.lfa_pred_thresh:
          ## Assemble a detection object
          det = dict()
          det['location'] = loc
          det['dim'] = dim
          det['rotation_y'] = rot_y
          if 'cat' in self.opt.snap_channels:
            det['cat'] = cls
          if 'cat_id' in self.opt.snap_channels:
            det['category_id'] = cls + 1 # map from index of class to id

          ## Compute distance threshold with 3D BB
          dist_thresh = get_dist_thresh_torch(None, None, dim, None, self.opt, phase,\
                                              loc, rot_y, device=device)
          
          ## Compute snap per detection
          if self.opt.snap_method == 'BEV':
            snap, frustum_bounds, nr_frustum_points, alternative_point, frustum_points = generate_snap_BEV_torch(pc_3d, det, dist_thresh, self.opt)
          elif self.opt.snap_method == 'proj':
            snap, frustum_bounds, nr_frustum_points, alternative_point, frustum_points = generate_snap_proj_torch(pc_3d, pc_snap_proj, det, \
                                                                        bbox, dist_thresh, trans, self.opt, calib)

          if nr_frustum_points > self.opt.limit_frustum_points:
            ## Forward pass of LFANet 
            # Predict optimal information from radar points inside frustum
            # Also need to return this since we need to compute a loss and call 
            # a backward pass on it.
            snap = snap.unsqueeze(0) # extend a dim for batch_size
            if self.opt.use_pointnet:
              out_val = self.convs(frustum_points[2:].unsqueeze(0))
            else:
              out_val = self.convs(snap)
            # Squeeze batches and scalar
            out_val = out_val.view(len(self.opt.pc_feat_lvl))

            # Transform depth with Eigen et. al to "euclidean depth"
            out_val[self.opt.pc_dep_index] = 1. / (out_val[self.opt.pc_dep_index].sigmoid() + 1e-6) - 1.

            with torch.no_grad():
              # Bound depth to frustum boundaries
              if self.opt.bound_lfa_to_frustum:
                out_val[self.opt.pc_dep_index] = torch.clamp(out_val[self.opt.pc_dep_index], min=frustum_bounds[0], max=frustum_bounds[1])

              # Project predicted velocity onto the radial axis of the shifted center point
              if self.opt.lfa_proj_vel_to_rad:
                vx = out_val[self.opt.pc_vx_index].unsqueeze(0)
                vz = out_val[self.opt.pc_vz_index].unsqueeze(0)
                loc_corr = loc.clone()
                # Shift 3D position
                loc_corr[2] = out_val[self.opt.pc_dep_index]
                # Project
                v_r = v_to_vrad_torch(torch.cat((vx,vz)), loc_corr[[0,2]])
                # Store
                out_val[self.opt.pc_vx_index] = v_r[0]
                out_val[self.opt.pc_vz_index] = v_r[1]
                
              # Debug LFANet
              if self.opt.debug > 1:
                # Plot radar points, bounding box and depth esimation
                fig, ax = plt.subplots()
                debug_lfa_frustum(det, out_val, snap, ax, self.opt, phase)
                plt.show()

              # Normalize depth for manual batch norming for sec heads
              if self.opt.normalize_depth:
                out_val[self.opt.pc_dep_index] /= self.opt.max_pc_depth
                
              ## Project features in bbox hm
              # Use either amodel or non_amodel for heatmap
              if not self.opt.lfa_not_use_amodel:
                bbox_hm = bbox_amodel
              else:
                bbox_hm = bbox
              # Compute center point from bbox 
              ct_bbox = torch.tensor(
                    [(bbox_hm[0] + bbox_hm[2]) / 2, (bbox_hm[1] + bbox_hm[3]) / 2], dtype=torch.float32)
              # Compute width and height from bbox
              w = bbox_hm[2] - bbox_hm[0]
              w_interval = self.opt.hm_to_box_ratio*(w)
              w_min = np.clip(int(ct_bbox[0] - w_interval/2.), 0, self.opt.output_w-3) # -3 to not index over pc_box_hm dim
              w_max = np.clip(int(ct_bbox[0] + w_interval/2.), 0, self.opt.output_w-3)

              h = bbox_hm[3] - bbox_hm[1]
              h_interval = self.opt.hm_to_box_ratio*(h)
              h_min = np.clip(int(ct_bbox[1] - h_interval/2.), 0, self.opt.output_h-3)
              h_max = np.clip(int(ct_bbox[1] + h_interval/2.), 0, self.opt.output_h-3)
              # Add filled bbox per detection to pc_box_hm
              for feat in self.opt.pc_feat_lvl:
                feat_ind = self.opt.pc_feat_channels[feat]
                pc_box_hm[img_ind, feat_ind,
                          h_min:h_max+1+1, 
                          w_min:w_max+1+1] = out_val[feat_ind] # twice +1 to handle 0 pixel wide and tall bbox
          
          elif (self.opt.limit_use_closest or self.opt.limit_use_vel) and nr_frustum_points > 0:
            # Get depth and velocity from alternative point
            dep = alternative_point[3]
            vx = alternative_point[4]
            vz = alternative_point[5]

            # Normalize depth for manual batch norming
            if self.opt.normalize_depth:
              depth = dep/self.opt.max_pc_depth
            else:
              depth = dep

            feature_values = dict.fromkeys(self.opt.pc_feat_lvl)
            feature_values['pc_dep'] = depth
            feature_values['pc_vx'] = vx
            feature_values['pc_vz'] = vz

            if self.opt.rcs_feature_hm:
              raise NotImplementedError

            ## Project features in bbox hm
            # Use either amodel or non_amodel for heatmap
            if not self.opt.lfa_not_use_amodel:
              bbox_hm = bbox_amodel
            else:
              bbox_hm = bbox
            # Compute center point from bbox 
            ct_bbox = torch.tensor(
                  [(bbox_hm[0] + bbox_hm[2]) / 2, (bbox_hm[1] + bbox_hm[3]) / 2], dtype=torch.float32, device=device)
            # Compute width and height from bbox
            w = bbox_hm[2] - bbox_hm[0]
            w_interval = self.opt.hm_to_box_ratio*(w)
            w_min = np.clip(int(ct_bbox[0] - w_interval/2.), 0, self.opt.output_w-3)
            w_max = np.clip(int(ct_bbox[0] + w_interval/2.), 0, self.opt.output_w-3)

            h = bbox_hm[3] - bbox_hm[1]
            h_interval = self.opt.hm_to_box_ratio*(h)
            h_min = np.clip(int(ct_bbox[1] - h_interval/2.), 0, self.opt.output_h-3)
            h_max = np.clip(int(ct_bbox[1] + h_interval/2.), 0, self.opt.output_h-3)

            # Map values uniformly in 2D bbox
            for feat in self.opt.pc_feat_lvl:
              if self.opt.limit_use_vel and feat not in ['pc_vx', 'pc_vz']:
                # Skip feats that are not velocity
                continue
              pc_box_hm[img_ind, self.opt.pc_feat_channels[feat],
                    h_min:h_max+1+1, 
                    w_min:w_max+1+1] = feature_values[feat] # twice +1 to handle 0 pixel wide and tall bbox

    return pc_box_hm


  def lfa_with_ann(self, batch, eval_frustum:EvalFrustum=None):
    device = batch['image'].device
    batch_size = batch['image'].shape[0]
    pc_lfa_feat = torch.zeros(batch_size, self.opt.max_objs, self.opt.lfa_pc_nr_feat,\
                  device=device) # no grad required if empty
    lfa_snap_mask = torch.zeros(batch_size, self.opt.max_objs, 1,\
                    device=device)
    pc_box_hm = torch.zeros_like(batch['pc_hm'])
                      
    all_nonzero = batch['mask'].nonzero()
    # Loop over images in batch
    for img_ind in range(batch['mask'].shape[0]):
      # Loop over all objects of the current image
      obj_inds = all_nonzero[all_nonzero[:,0] == img_ind][:,1]
      for obj_ind in obj_inds:
        snap = batch['snaps'][img_ind][obj_ind]
        nr_frustum_points = batch['nr_frustum_points'][img_ind][obj_ind]
        ## Forward pass of LFANet 
        # Predict optimal information from radar points inside frustum
        # Also need to return this since we need to compute a loss and call 
        # a backward pass on it.
        if nr_frustum_points > self.opt.limit_frustum_points:
          snap = snap.unsqueeze(0) # extend a dim for batch_size
          out = self.convs(snap)
          # Squeeze batches and scalar because we dont calc output of network over all batches
          out = out.view(self.opt.lfa_pc_nr_feat)
          # Transform depth with Eigen et. al
          out[self.opt.pc_dep_index] = 1. / (out[self.opt.pc_dep_index].sigmoid() + 1e-6) - 1.
          # Store for loss calculation
          pc_lfa_feat[img_ind, obj_ind] = out
          # Store mask 
          lfa_snap_mask[img_ind, obj_ind] = 1
          
          # Bound z output by frustum boundaries if wanted. This does not influece loss calculation.
          if self.opt.bound_lfa_to_frustum:
            frustum_bounds = batch['frustum_bounds'][img_ind][obj_ind]
            out[self.opt.pc_dep_index] = torch.clamp(out[self.opt.pc_dep_index], min=frustum_bounds[0], max=frustum_bounds[1])
            
          # Debug LFANet
          if self.opt.debug > 1:
            # Create obj from image
            obj = dict()
            obj['dim'] = batch['dim'][img_ind][obj_ind]
            obj['location'] = batch['location'][img_ind][obj_ind]
            obj['rotation_y'] = batch['rotation_y'][img_ind][obj_ind]
            if 'cat' in self.opt.snap_channels:
              obj['cat'] = batch['cat'][img_ind][obj_ind]
            if 'cat_id' in self.opt.snap_channels:
              obj['category_id'] = batch['cat'][img_ind][obj_ind] + 1
            
            if self.opt.debug > 2:
              # Compare snaps

              # Compute distance threshold with 3D BB 
              dist_thresh = get_dist_thresh_torch(None, None, obj['dim'], None, self.opt, 'train', \
                                                  obj['location'], obj['rotation_y'], device)

              # Differ between different snap generation methods
              pc_3d = copy.deepcopy(batch['pc_3d'][img_ind][:,0:batch['pc_N'][img_ind]])

              if self.opt.snap_method == 'proj':
                snap_BEV, _, _, _, _ = generate_snap_BEV_torch(pc_3d, obj, dist_thresh, self.opt)       
                snap_proj = snap.squeeze()
              elif self.opt.snap_method == 'BEV':
                snap_proj, _, _, _, _ = generate_snap_proj_torch(pc_3d, batch['pc_snap_proj'][img_ind], obj, batch['bbox'][img_ind][obj_ind], dist_thresh, \
                                                                  batch['trans_original'][img_ind], self.opt, batch['calib'][img_ind])  
                snap_BEV = snap.squeeze()

              # snap1 = snap.squeeze()
              snap_BEV = np.around(snap_BEV.detach().cpu().numpy(),2)
              snap_proj = np.around(snap_proj.detach().cpu().numpy(),2)

              # Only plot something if snaps do not match perfectly
              if not np.all(snap_BEV==snap_proj):
                idxxx = np.where(snap_BEV!=snap_proj)[0][0]
                print(f'Channel {self.opt.snap_channels[idxxx]} plotted')
                snap_BEV_norm = snap_BEV[idxxx,:,::-1] + np.abs(np.min((0, snap_BEV[idxxx,:,:].min())))
                snap_proj_norm = snap_proj[idxxx,:,::-1] + np.abs(np.min((0, snap_proj[idxxx,:,:].min())))
                snap_BEV_norm = np.array((snap_BEV_norm * (1/(snap_BEV_norm.max()+1e-9))),dtype=np.float32)
                snap_proj_norm = np.array((snap_proj_norm * (1/(snap_proj_norm.max()+1e-9))),dtype=np.float32)
            
                cv2.imshow('BEV', cv2.resize(snap_BEV_norm.T, (640,640), interpolation=cv2.INTER_AREA))
                cv2.imshow('proj', cv2.resize(snap_proj_norm.T, (640,640), interpolation=cv2.INTER_AREA))

                cv2.waitKey()
            
            obj['velocity'] = batch['velocity'][img_ind][obj_ind]
            # Plot radar points, bounding box and depth esimation 
            fig, ax = plt.subplots()
            debug_lfa_frustum(obj, out, snap, ax, self.opt, 'train') # (phase might be wrong. This fnc is also called in val)
            plt.show()
          
          # Forward LFA output to sec heads by filling in the pc box heatmap
          if self.opt.lfa_forward_to_sec:
            # Normalize depth for manual batch norming for sec heads
            if self.opt.normalize_depth:
              out[self.opt.pc_dep_index] /= self.opt.max_pc_depth
              
            ## Project features in bbox hm 
            # Compute center point from bbox
            if not self.opt.lfa_not_use_amodel:
              bbox_hm = batch['bbox_amodel'][img_ind][obj_ind]
            else:
              bbox_hm = batch['bbox'][img_ind][obj_ind]
            # Compute center point from bbox 
            ct_bbox = torch.tensor(
                  [(bbox_hm[0] + bbox_hm[2]) / 2, (bbox_hm[1] + bbox_hm[3]) / 2], dtype=torch.float32)
            # Compute width and height from bbox
            w = bbox_hm[2] - bbox_hm[0]
            w_interval = self.opt.hm_to_box_ratio*(w)
            w_min = np.clip(int(ct_bbox[0] - w_interval/2.), 0, self.opt.output_w-3)
            w_max = np.clip(int(ct_bbox[0] + w_interval/2.), 0, self.opt.output_w-3)

            h = bbox_hm[3] - bbox_hm[1]
            h_interval = self.opt.hm_to_box_ratio*(h)
            h_min = np.clip(int(ct_bbox[1] - h_interval/2.), 0, self.opt.output_h-3)
            h_max = np.clip(int(ct_bbox[1] + h_interval/2.), 0, self.opt.output_h-3)
            # Add filled bbox per detection to pc_box_hm
            for feat in self.opt.pc_feat_lvl:
              feat_ind = self.opt.pc_feat_channels[feat]
              pc_box_hm[img_ind, feat_ind,
                        h_min:h_max+1+1, 
                        w_min:w_max+1+1] = out[feat_ind] # twice +1 to handle 0 pixel wide and tall bbox

        elif self.opt.lfa_forward_to_sec and (self.opt.limit_use_closest or self.opt.limit_use_vel) and nr_frustum_points > 0:
          alternative_point = batch['alternative_point'][img_ind][obj_ind]
          # Get depth and velocity from alternative point
          dep = alternative_point[3]
          vx = alternative_point[4]
          vz = alternative_point[5]

          # Normalize depth for manual batch norming
          if self.opt.normalize_depth:
            depth = dep/self.opt.max_pc_depth
          else:
            depth = dep

          feature_values = dict.fromkeys(self.opt.pc_feat_lvl)
          feature_values['pc_dep'] = depth
          feature_values['pc_vx'] = vx
          feature_values['pc_vz'] = vz

          if self.opt.rcs_feature_hm:
            raise NotImplementedError

          ## Project features in bbox hm 
          # Compute center point from bbox
          if not self.opt.lfa_not_use_amodel:
            bbox_hm = batch['bbox_amodel'][img_ind][obj_ind]
          else:
            bbox_hm = batch['bbox'][img_ind][obj_ind]
          # Compute center point from bbox 
          ct_bbox = torch.tensor(
                [(bbox_hm[0] + bbox_hm[2]) / 2, (bbox_hm[1] + bbox_hm[3]) / 2], dtype=torch.float32)
          # Compute width and height from bbox
          w = bbox_hm[2] - bbox_hm[0]
          w_interval = self.opt.hm_to_box_ratio*(w)
          w_min = np.clip(int(ct_bbox[0] - w_interval/2.), 0, self.opt.output_w-3)
          w_max = np.clip(int(ct_bbox[0] + w_interval/2.), 0, self.opt.output_w-3)

          h = bbox_hm[3] - bbox_hm[1]
          h_interval = self.opt.hm_to_box_ratio*(h)
          h_min = np.clip(int(ct_bbox[1] - h_interval/2.), 0, self.opt.output_h-3)
          h_max = np.clip(int(ct_bbox[1] + h_interval/2.), 0, self.opt.output_h-3)

          # Map values uniformly in 2D bbox
          for feat in self.opt.pc_feat_lvl:
            if self.opt.limit_use_vel and feat not in ['pc_vx', 'pc_vz']:
              # Skip feats that are not velocity
              continue
            pc_box_hm[img_ind, self.opt.pc_feat_channels[feat],
                  h_min:h_max+1+1, 
                  w_min:w_max+1+1] = feature_values[feat] # twice +1 to handle 0 pixel wide and tall bbox
  

        if self.opt.eval_frustum > 0 and nr_frustum_points > self.opt.limit_frustum_points:
          ann_object = dict()
          ann_object['dim'] = batch['dim'][img_ind][obj_ind]
          ann_object['location'] = batch['location'][img_ind][obj_ind]
          ann_object['rotation_y'] = batch['rotation_y'][img_ind][obj_ind]
          if 'cat' in self.opt.snap_channels:
            ann_object['cat'] = batch['cat'][img_ind][obj_ind]
          if 'cat_id' in self.opt.snap_channels:
            ann_object['category_id'] = batch['cat'][img_ind][obj_ind] + 1
          ann_object['velocity_cam'] = batch['velocity'][img_ind][obj_ind]
          # Compute dist thresh
          dist_thresh = get_dist_thresh_torch(None, None, ann_object['dim'], None, self.opt, phase='train', location=ann_object['location'], rot_y = ann_object['rotation_y'])

          # Differ between different snap generation methods
          pc_3d = copy.deepcopy(batch['pc_3d'][img_ind][:,0:batch['pc_N'][img_ind]])
          
          # Generate snapshot of object (frustum points at the moment not precomputed)
          if self.opt.snap_method == 'BEV':
            _, _, nr_frustum_points_eval, _, frustum_points = generate_snap_BEV_torch(pc_3d, ann_object, dist_thresh, self.opt)
          elif self.opt.snap_method == 'proj':
            _, _, nr_frustum_points_eval, _, frustum_points  = generate_snap_proj_torch(pc_3d, batch['pc_snap_proj'][img_ind], ann_object, batch['bbox'][img_ind][obj_ind], dist_thresh, \
                                                                  batch['trans_original'][img_ind], self.opt, batch['calib'][img_ind])
          
          if not frustum_points == None:

            # retrieve original depth if normalized
            if self.opt.normalize_depth and self.opt.lfa_forward_to_sec:
                out[self.opt.pc_dep_index] *= self.opt.max_pc_depth

            # Collect parameters for analysis
            frustum_points = frustum_points.detach().cpu().numpy()
            r_star = np.zeros((8,1)) # artificial radar point
            r_star[2,:] = ann_object['location'][0].detach().cpu().numpy() # x
            
            r_star[3,:] = out[self.opt.pc_dep_index].detach().cpu().numpy() # depth
            r_star[4,:] = out[1].detach().cpu().numpy() # vx
            r_star[5,:] = out[2].detach().cpu().numpy() # vz
            # x, rcs, dts are not predicted but also not used in analyzing (rcs is only used for plotting)
            frustum_points= np.append(frustum_points, r_star, axis=1)
            pc_pos_x_match = frustum_points[2,:]
            pc_dep_match = frustum_points[3,:]
            pc_vx_match = frustum_points[4,:]
            pc_vz_match = frustum_points[5,:]
            pc_rcs_match = frustum_points[6,:]
            pc_dts_match = frustum_points[7,:]
            idx_selection = frustum_points.shape[1]-1 # take last point
            for e in ann_object:
              ann_object[e] = ann_object[e].detach().cpu().numpy()
            dist_thresh = dist_thresh.detach().cpu().numpy()

            eval_frustum.analyze_frustum_association(pc_pos_x_match, pc_dep_match,
                                                    pc_vx_match, pc_vz_match,                                                                                                                       
                                                    pc_rcs_match,
                                                    pc_dts_match,
                                                    idx_selection,
                                                    ann_object,
                                                    frustum_dist_thresh=dist_thresh,
                                                    )

    if not self.opt.lfa_forward_to_sec:
      # Use feats of annotation instead of LFA outputs in pc bb hm
      pc_box_hm = batch['pc_box_hm']

    return pc_box_hm, pc_lfa_feat, lfa_snap_mask

class MapChannels(nn.Module):
  def __init__(self, opt, last_channel):
      kernel_size = 1
      super(MapChannels, self).__init__()
      # Additional Conv layer to match radar channels to img channels of feature map
      self.maplayer = Sequential(
          Conv2d(opt.lfa_pc_nr_feat, last_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
          ReLU()
      )
      print("Mapping to Feature Map: ", self.maplayer)

  def match_channels(self, pc_box_hm):
    """
    Function to apply self.maplayer function which includes a convolutional
    layer that matches the radar feature channels to the channels in the
    image-based feature map, which is by default set to last_channel = 64 if num_layers == 34 else 128 due to the fixed structure of this DLA implementation.

    :param pc_box_hm: torch tensor (batch_size, pc_nr_feat, out_w, out_h)
                      point cloud BB heatmap filled with radar features
    
    :return: pc_box_hm torch tensor (batch_size, 64, out_w, out_h)
                      point cloud BB heatmap filled with radar features
    """
    return self.maplayer(pc_box_hm)

def set_pc_box_hm(pc_box_hm, bbox, bbox_amodel, dep, gt_rad_vel, rcs, nr_frustum_points, alternative_point, opt):
  """
  Compute GT pc BB hm to forward to secondary heads in LFANet. (training and gt val)
  It is computed differently than in Nabati's CF because we need the gt values of the object and
  not of the pc.
  If there are no points inside the object's frustum, don't fill anything new into pc_box_hm
  :param pc_box_hm: np_array [C x H x W] Image-like array to store the 2D BB with their feature
                    values. Annotations are sorted reverse depth-wise to allow closer BBs to
                    overwrite farther ones. 
  :param pc_3d: 3d point cloud
  :param ann: dict() annotation object
  :param bbox: np_array [4] GT 2D BB in image
  :param dep: float GT depth of center point. Scaled to match scale factor of data augmentation.
  :param gt_rad_vel: np_array [2] Projected radial GT velocity of center point in camera coordinates
  :param alternative_point: alternative to artificial radar point from LFANet
  :param opt: dict Options
  """
  if nr_frustum_points > opt.limit_frustum_points:
    # Everything in numpy (not torch)
    # Normalize depth for manual batch norming
    if opt.normalize_depth:
      depth = dep/opt.max_pc_depth

    # Save feature values in feature_values
    feature_values = dict.fromkeys(opt.pc_feat_lvl)
    feature_values['pc_dep'] = depth
    feature_values['pc_vx'] = gt_rad_vel[0]
    feature_values['pc_vz'] = gt_rad_vel[1]

    if opt.rcs_feature_hm and rcs != None:
      feature_values['pc_rcs'] = rcs


    ## Project features in bbox hm 
    # Compute center point from bbox
    if not opt.lfa_not_use_amodel:
      bbox_hm = bbox_amodel
    else:
      bbox_hm = bbox
    # Compute center point from bbox 
    ct_bbox = np.array(
          [(bbox_hm[0] + bbox_hm[2]) / 2, (bbox_hm[1] + bbox_hm[3]) / 2], dtype=np.float32)
    # Compute width and height from bbox
    w = bbox_hm[2] - bbox_hm[0]
    w_interval = opt.hm_to_box_ratio*(w)
    w_min = np.clip(int(ct_bbox[0] - w_interval/2.), 0, opt.output_w-3)
    w_max = np.clip(int(ct_bbox[0] + w_interval/2.), 0, opt.output_w-3)

    h = bbox_hm[3] - bbox_hm[1]
    h_interval = opt.hm_to_box_ratio*(h)
    h_min = np.clip(int(ct_bbox[1] - h_interval/2.), 0, opt.output_h-3)
    h_max = np.clip(int(ct_bbox[1] + h_interval/2.), 0, opt.output_h-3)

    # Map values uniformly in 2D bbox
    for feat in opt.pc_feat_lvl:
      pc_box_hm[opt.pc_feat_channels[feat],
            h_min:h_max+1+1, 
            w_min:w_max+1+1] = feature_values[feat] # twice +1 to handle 0 pixel wide and tall bbox
  
  elif (opt.limit_use_closest or opt.limit_use_vel) and nr_frustum_points > 0:
    # Get depth and velocity from alternative point
    dep = alternative_point[3]
    vx = alternative_point[4]
    vz = alternative_point[5]

    # Normalize depth for manual batch norming
    if opt.normalize_depth:
      depth = dep/opt.max_pc_depth
    else:
      depth = dep

    feature_values = dict.fromkeys(opt.pc_feat_lvl)
    feature_values['pc_dep'] = depth
    feature_values['pc_vx'] = vx
    feature_values['pc_vz'] = vz

    if opt.rcs_feature_hm and rcs != None:
      raise NotImplementedError

    ## Project features in bbox hm 
    # Compute center point from bbox
    if not opt.lfa_not_use_amodel:
      bbox_hm = bbox_amodel
    else:
      bbox_hm = bbox
    # Compute center point from bbox 
    ct_bbox = np.array(
          [(bbox_hm[0] + bbox_hm[2]) / 2, (bbox_hm[1] + bbox_hm[3]) / 2], dtype=np.float32)
    # Compute width and height from bbox
    w = bbox_hm[2] - bbox_hm[0]
    w_interval = opt.hm_to_box_ratio*(w)
    w_min = np.clip(int(ct_bbox[0] - w_interval/2.), 0, opt.output_w-3)
    w_max = np.clip(int(ct_bbox[0] + w_interval/2.), 0, opt.output_w-3)

    h = bbox_hm[3] - bbox_hm[1]
    h_interval = opt.hm_to_box_ratio*(h)
    h_min = np.clip(int(ct_bbox[1] - h_interval/2.), 0, opt.output_h-3)
    h_max = np.clip(int(ct_bbox[1] + h_interval/2.), 0, opt.output_h-3)

    # Map values uniformly in 2D bbox
    for feat in opt.pc_feat_lvl:
      if opt.limit_use_vel and feat not in ['pc_vx', 'pc_vz']:
        # Skip feats that are not velocity
        continue
      pc_box_hm[opt.pc_feat_channels[feat],
            h_min:h_max+1+1, 
            w_min:w_max+1+1] = feature_values[feat] # twice +1 to handle 0 pixel wide and tall bbox 