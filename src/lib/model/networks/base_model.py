from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..utils import _topk, _tranpose_and_gather_feat
from utils.pointcloud import generate_pc_box_hm
from utils.ddd_utils import project_to_image_torch
from model.networks.lfanet import LFANet, MapChannels
import torch
from torch import nn
import cv2
import numpy as np

def fill_fc_weights(layers):
  """
  Initiate bias in layers to 0
  """
  for m in layers.modules():
    if isinstance(m, nn.Conv2d):
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)


class BaseModel(nn.Module):
  def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
    super(BaseModel, self).__init__()
    self.opt = opt
    if opt is not None and opt.head_kernel != 3:
      head_kernel = opt.head_kernel
    else:
      head_kernel = 3
    print('Using head kernel:', head_kernel)
    
    self.num_stacks = num_stacks
    self.heads = heads
    self.secondary_heads = opt.secondary_heads

    # self.early_fusion_bn = nn.BatchNorm2d(len(self.opt.early_fusion_channels))
    
    # Last channels before head module, i.e. last nr of channels in backbone
    last_channels = {head: last_channel for head in heads}
    # Extra output channels for secondary (pointcloud) head
    for head in self.secondary_heads:
      if opt.lfa_match_channels:
        # PC BB heatmap adds the same nr of channels as Feature Map
        last_channels[head] = last_channel*2
      else:
        # PC BB heatmap adds radar features
        last_channels[head] = last_channel+len(opt.pc_feat_lvl)
    
    for head in self.heads:
      classes = self.heads[head]
      head_conv = head_convs[head]
      if len(head_conv) > 0: # if a head layer has no channels, skip it
        # Create head layers
        conv = nn.Conv2d(last_channels[head], head_conv[0], 
                          kernel_size=head_kernel, 
                          padding=head_kernel // 2, bias=True) # padding set such that there is no down-/upscaling
        convs = [conv]
        
        # Add more layers for sec heads with corresponding in and out 
        # channels
        if opt.extended_head_arch:
          # Bugfix: As proposed in paper
          for k in range(1, len(head_conv)):
              convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                           kernel_size=head_kernel, 
                           padding=head_kernel // 2, bias=True))
        else:
          # CenterFusion
          for k in range(1, len(head_conv)):
              convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                           kernel_size=1, bias=True))
          
        # Last head layer with nr of classes as channels an no kernel
        out = nn.Conv2d(head_conv[-1], classes, kernel_size=1, 
                        stride=1, padding=0, bias=True)
        
        # Lastly combine all layers 
        if opt.bn_in_head_arch:
          # Use BN in heads
          if len(convs) == 1:
            fc = nn.Sequential(
              conv, nn.BatchNorm2d(head_conv[0]),
              nn.ReLU(inplace=True), 
              out)
          elif len(convs) == 2:
            fc = nn.Sequential(
              convs[0], nn.BatchNorm2d(head_conv[0]), 
              nn.ReLU(inplace=True), 
              convs[1], nn.BatchNorm2d(head_conv[1]),
              nn.ReLU(inplace=True),
              out)
          elif len(convs) == 3:
            fc = nn.Sequential(
              convs[0], nn.BatchNorm2d(head_conv[0]),
              nn.ReLU(inplace=True), 
              convs[1], nn.BatchNorm2d(head_conv[1]),
              nn.ReLU(inplace=True), 
              convs[2], nn.BatchNorm2d(head_conv[2]),
              nn.ReLU(inplace=True), 
              out)
          elif len(convs) == 4:
            fc = nn.Sequential(
              convs[0], nn.BatchNorm2d(head_conv[0]),
              nn.ReLU(inplace=True), 
              convs[1], nn.BatchNorm2d(head_conv[1]),
              nn.ReLU(inplace=True),
              convs[2], nn.BatchNorm2d(head_conv[2]), 
              nn.ReLU(inplace=True),
              convs[3], nn.BatchNorm2d(head_conv[3]), 
              nn.ReLU(inplace=True),
              out)
          else: 
            raise NotImplementedError('Add more conv layers in the head as desired.')
        else:
          # CenterFusion / CenterNet
          if len(convs) == 1:
            fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
          elif len(convs) == 2:
            fc = nn.Sequential(
              convs[0], nn.ReLU(inplace=True), 
              convs[1], nn.ReLU(inplace=True), out)
          elif len(convs) == 3:
            fc = nn.Sequential(
                convs[0], nn.ReLU(inplace=True), 
                convs[1], nn.ReLU(inplace=True), 
                convs[2], nn.ReLU(inplace=True), out)
          elif len(convs) == 4:
            fc = nn.Sequential(
                convs[0], nn.ReLU(inplace=True), 
                convs[1], nn.ReLU(inplace=True), 
                convs[2], nn.ReLU(inplace=True), 
                convs[3], nn.ReLU(inplace=True), out)
          else: 
            raise NotImplementedError('Add more conv layers in the head as desired.')
        # Bias
        if 'hm' in head:
          # Initiate bias of the output layer for the heatmap head to a prior bias
          fc[-1].bias.data.fill_(opt.prior_bias)
        else: 
          # Initiate bias to 0
          fill_fc_weights(fc)
      else:
        # If there is no head layer, at least convert from last backbone channels to needed 'classes' for regression
        fc = nn.Conv2d(last_channels[head], classes, 
            kernel_size=1, stride=1, padding=0, bias=True)

        # Bias
        if 'hm' in head:
          # Initiate bias of the output layer for the heatmap head to a prior bias
          fc.bias.data.fill_(opt.prior_bias)
        else: 
          # Initiate bias to 0
          fill_fc_weights(fc)

      self.__setattr__(head, fc) # create function calls for forward pass
    
    if opt.use_lfa:
      # Set up LFA 
      self.lfa = LFANet(opt)
    
    if opt.lfa_match_channels:
      # Create mapping layer
      self.map = MapChannels(opt, last_channel)
    
  def img2feats(self, x):
    # The function of the child is called
    raise NotImplementedError
  
  def imgpre2feats(self, x, pre_img=None, pre_hm=None):
    # The function of the child is called
    raise NotImplementedError

  def forward(self, batch, eval_frustum=None):
    """ 
    Forward pass for all heads.  
    :param batch: Current Batch (dict) with all the informatin necessaray provided, e.g.
      :key pc_hm is the heatmap or better the filtered radar pillars.
      :key image: img data as input for the backbone
      :key pc_hm_add: not used as training data but to calc. dist_thresh when using dist approach e.g.
      :key trans_original: necessary to correctly compute dist thres (back-trafo to image size for camera matrix)"""
    

    # EARLY FUSION 
    if self.opt.use_early_fusion:
      # Concatenate the input tensors for the backbone:
      # RGB image + pc_ef including the radar channels projected to the image plane
      early_fusion_norm = torch.clone(batch['pc_ef'])

      if self.opt.normalize_depth and 'z' in self.opt.early_fusion_channels:
        # Normalize depth values by dividing through self.opt.max_radar_distance
        early_fusion_norm[:,self.opt.early_fusion_channels.index('z'),:,:] /= self.opt.max_pc_depth 
      
      if 'dts' in self.opt.early_fusion_channels:
        # Normalize timestamps
        early_fusion_norm[:,self.opt.early_fusion_channels.index('dts'),:,:] /= self.opt.dts_norm 

        # Batch-normalize input features for early fusion 
        # - deprecated since questionable for velocity especially
        # early_fusion_norm = self.early_fusion_bn(batch['pc_ef'])

      x = torch.cat((batch['image'], early_fusion_norm), 1)

    else:    
      # Input to backbone is RGB image
      x = batch['image']

    ## extract features from image through backbone + neck (upsampled to 200x112 by default)
    feats = self.img2feats(x)
    out = []
    
    for s in range(self.num_stacks):
      z = {}
      
      ## Run the first stage heads
      for head in self.heads:
        if head not in self.secondary_heads:
          # Evaluate PRIMARY heads and fill predictions
          z[head] = self.__getattr__(head)(feats[s])

      if self.opt.use_sec_heads:
        ## get pointcloud bounding box heatmap 
        if not self.training:
          if not self.opt.eval_with_gt:
            if self.opt.disable_frustum:
              pc_box_hm = batch.get('pc_hm', None)
              if self.opt.normalize_depth:
                pc_box_hm[self.opt.pc_feat_channels['pc_dep']] /= self.opt.max_pc_depth
            else:
              ## FRUSTUM CREATION WITH PREDICTION FOR VALIDATION/TESTING ##
              # contains expansion ratio i.e. delta
              if self.opt.use_lfa:
                # Use the LFA heatmap generation
                pc_box_hm = self.lfa.generate_lfa_pc_box_hm_torch(batch, z, 'val')
                z['pc_box_hm'] = pc_box_hm
              else:
                # Use Nabatis heatmap generation
                pc_hm = batch.get('pc_hm', None)
                pc_hm_add = batch.get('pc_hm_add', None)
                pc_box_hm = generate_pc_box_hm(z, pc_hm, pc_hm_add, 
                                               batch['calib'], 
                                               self.opt, batch.get('trans_original', None))
          else: # evaluate with gt
            pc_box_hm = batch['pc_box_hm']

        else: # if training
          if self.opt.use_lfa and not self.opt.train_with_gt:
            if self.opt.lfa_with_ann:
              # Use gt object as input to LFANet
              pc_box_hm, pc_lfa_feat, lfa_snap_mask = self.lfa.lfa_with_ann(batch, eval_frustum=eval_frustum)
              z['pc_lfa_feat'] = pc_lfa_feat
              z['lfa_snap_mask'] = lfa_snap_mask
            else:
              # Use prediction of prim head as input to LFANet
              pc_box_hm = self.lfa.generate_lfa_pc_box_hm_torch(batch, z, 'train')
              z['pc_box_hm'] = pc_box_hm
          else:
            pc_box_hm = batch.get('pc_box_hm', None)
            
        if self.opt.debug > 0:
          z['pc_box_hm'] = pc_box_hm

        ## Addtional conv layer to match the number of channels of radar to image-based feature map 
        if self.opt.lfa_match_channels:
          pc_box_hm = self.map.match_channels(pc_box_hm)

        ## Run the second stage heads
        sec_feats = [feats[s], pc_box_hm]  ### FUSION OF IMG FEATURES AND RADAR DATA [pc_dep, pc_vx, pc_vz] ###
        sec_feats = torch.cat(sec_feats, 1) # Concatenate in first dim
        for head in self.secondary_heads:
          # Evaluate SECONDARY heads and fill predictions
          z[head] = self.__getattr__(head)(sec_feats)

      out.append(z)

    return out

