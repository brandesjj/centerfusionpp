from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import copy
import numpy as np
from progress.bar import Bar

from model.data_parallel import DataParallel
from model.model import eval_layers
from utils.utils import AverageMeter

from model.losses import FastFocalLoss, LFALoss, RegWeightedL1Loss, \
                         DepthLoss, BinRotLoss, WeightedBCELoss
from model.decode import fusion_decode
from model.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import generic_post_process

import cv2
class GenericLoss(torch.nn.Module):
  def __init__(self, opt):
    super(GenericLoss, self).__init__()
    self.crit = FastFocalLoss()
    self.crit_reg = RegWeightedL1Loss()
    if 'rot' in opt.heads:
      self.crit_rot = BinRotLoss()
    if 'nuscenes_att' in opt.heads:
      self.crit_nuscenes_att = WeightedBCELoss()
    self.opt = opt
    self.crit_dep = DepthLoss()
    if opt.use_lfa:
      self.crit_lfa = LFALoss()

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    if 'dep_sec' in output and self.opt.sigmoid_dep_sec:
      output['dep_sec'] = 1. / (output['dep_sec'].sigmoid() + 1e-6) - 1.
    # NOTE: depth of LFANet is already transformed
    return output

  def forward(self, outputs, batch, phase='train'):
    """
    :param outputs: tensor Output of network in stacks.
    :param batch: dict Batch object
    :param phase: String Phase is either 'train' or 'val'. For 'val' 
                  always calculate loss but in 'train' only if grad is
                  also required. The loss is skipped in training to save
                  runtime.
    
    :return: losses['tot'] scalar tensor Sum of all single losses that
             require grad.
    :return: losses dict Whole losses dict with all single losses logged.
    """
    opt = self.opt
    losses = {head: torch.tensor(0, device=opt.device, dtype=torch.float32) for head in opt.heads}
    losses.update({'pc_lfa_feat': torch.tensor(0, device=opt.device, dtype=torch.float32)})

    for s in range(opt.num_stacks):
      output = outputs[s]
      output = self._sigmoid_output(output)
      if 'hm' in output and (output['hm'].requires_grad \
                             or phase == 'val'):
        losses['hm'] += self.crit(
          output['hm'], batch['hm'], batch['ind'], 
          batch['mask'], batch['cat']) / opt.num_stacks
      if 'dep' in output and (output['dep'].requires_grad \
                              or phase == 'val'):
        losses['dep'] += self.crit_dep(
          output['dep'], batch['dep'], batch['ind'], 
          batch['dep_mask'], batch['cat']) / opt.num_stacks
      regression_heads = [
        'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodel', 'hps', 
        'dim', 'amodel_offset', 'velocity']

      for head in regression_heads:
        if head in output and (output[head].requires_grad \
                               or phase == 'val'):
          losses[head] += self.crit_reg(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks

      if 'hm_hp' in output and (output['hm_hp'].requires_grad \
                                or phase == 'val'):
        losses['hm_hp'] += self.crit(
          output['hm_hp'], batch['hm_hp'], batch['hp_ind'], 
          batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
        if 'hp_offset' in output and (output['hp_offset'].requires_grad \
                                      or phase == 'val'):
          losses['hp_offset'] += self.crit_reg(
            output['hp_offset'], batch['hp_offset_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
        
      if 'rot' in output and (output['rot'].requires_grad \
                              or phase == 'val'):
        losses['rot'] += self.crit_rot(
          output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
          batch['rotres'], opt) / opt.num_stacks        

      if 'nuscenes_att' in output and (output['nuscenes_att'].requires_grad\
                                       or phase == 'val'):
        losses['nuscenes_att'] += self.crit_nuscenes_att(
          output['nuscenes_att'], batch['nuscenes_att_mask'],
          batch['ind'], batch['nuscenes_att']) / opt.num_stacks
      
      if 'dep_sec' in output and (output['dep_sec'].requires_grad \
                                  or phase == 'val'):
        losses['dep_sec'] += self.crit_dep(
          output['dep_sec'], batch['dep'], batch['ind'], 
          batch['dep_mask'], batch['cat']) / opt.num_stacks
      
      if 'rot_sec' in output and (output['rot_sec'].requires_grad \
                                  or phase == 'val'):
        losses['rot_sec'] += self.crit_rot(
          output['rot_sec'], batch['rot_mask'], batch['ind'], batch['rotbin'],
          batch['rotres'], opt) / opt.num_stacks

      if opt.use_lfa:
        if 'pc_lfa_feat' in output and (output['pc_lfa_feat'].requires_grad):
          if phase == 'train' and opt.lfa_with_ann:
            # Also calculate loss with targets in validation when training 
            # standalone LFANet
            losses['pc_lfa_feat'] += self.crit_lfa(
              output['pc_lfa_feat'], batch['pc_lfa_feat'], snap_mask=output['lfa_snap_mask'],
              mask=batch['lfa_mask'], weights=opt.lfa_weights.to(device=opt.device)) / opt.num_stacks
        elif 'pc_box_hm' in output and not opt.lfa_skip_loss:
            # Calculate loss by comparing the pred and gt pc_box_hm since predictions cannot be compared directly
            # to targets anymore. They don't necessarily correspond to the
            # same object!
            # Use L1 loss
            # This loss is quite large and hard to interpret since every pixel in the hm are compared. It is recommended to skip it with --lfa_skip_loss if not needed.
            losses['pc_lfa_feat'] += self.crit_lfa(
              output['pc_box_hm'], batch['pc_box_hm'], lfa_with_ann=False) / opt.num_stacks
    losses['tot'] = 0
    for head in opt.heads:
      losses['tot'] += opt.weights[head] * losses[head]
    if 'pc_lfa_feat' in output and (output['pc_lfa_feat'].requires_grad \
                                    or phase == 'val') and not opt.lfa_skip_loss:
      losses['tot'] += opt.weights['pc_lfa_feat'] * losses['pc_lfa_feat']
    
    return losses['tot'], losses


class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss, opt):
    super(ModelWithLoss, self).__init__()
    self.opt = opt
    self.model = model
    self.loss = loss
  
  def forward(self, batch, phase, eval_frustum=None):

    outputs = self.model(batch, eval_frustum=eval_frustum)
    
    ## Compute losses
    loss, loss_stats = self.loss(outputs, batch, phase)

    # only return the last stack (per default only one stack)
    return outputs[-1], loss, loss_stats 


class Trainer(object):
  def __init__(self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss, opt)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      print(f'#######################################################\nDevice IDs used for DataParallel initialization')
      print(f'Chunk sizes are {chunk_sizes}')
      print('#######################################################')
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      print('Only 1 GPU used for training.')
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader, eval_frustum=None):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train() # Just sets the whole model in "training" mode
      if self.opt.set_eval_layers:
        if len(self.opt.gpus) > 1:
          eval_layers(model_with_loss.module.model, self.opt) # Set specified layers to eval mode
        else:
          eval_layers(model_with_loss.model, self.opt) # Set specified layers to eval mode
    else:
      # Validation phase
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats \
                      if l == 'tot' or opt.weights[l] > 0}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    
        # ITERATE OVER BATCHES
    print("-"*20, " Epoch: ", epoch, " ", "-"*20)
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      for k in batch:
        if k != 'meta':
          # Convert numpy arrays to torch tensors
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)  
      
      # run one iteration 
      output, loss, loss_stats = model_with_loss(batch, phase, eval_frustum=eval_frustum)
      
      # backpropagate and step optimizer
      loss = loss.mean()
      if phase == 'train':
        if loss.requires_grad:
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
        else:
          pass

      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['image'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
        '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0: # If not using progress bar
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      # Create visualization for debug with output
      if opt.debug > 0:
        self.debug(batch, output, iter_id, dataset=data_loader.dataset)
      
      # generate detections for evaluation
      if (phase == 'val' and (opt.run_dataset_eval or opt.eval)):
        # 3D box encoder to put together predictions
        dets = fusion_decode(output, K=opt.K, opt=opt)

        for k in dets:
          dets[k] = dets[k].detach().cpu().numpy()
        
        assert len(batch['meta']) == 1, "Batch size should be 1 for validation"

        meta = batch['meta'][0]
        calib = meta['calib'].detach().numpy() if 'calib' in meta else None
        
        dets = generic_post_process(opt, dets, 
          meta['c'].cpu().numpy(), meta['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib) 

        # merge results
        result = []
        for i in range(len(dets[0])): # only one batch since validation -> 0 ind of dets
          # Loop over top K predictions and only store predictions over a specific certainty value
          if dets[0][i]['score'] > self.opt.out_thresh and all(dets[0][i]['dim'] > 0):
            result.append(dets[0][i])

        img_id = batch['meta'][0]['img_id'].numpy().astype(np.int32)[0]
        results[img_id] = result

 
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results


  def _get_losses(self, opt):
    loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
      'hp_offset', 'dep', 'dep_sec', 'dim', 'rot', 'rot_sec',
      'amodel_offset', 'ltrb_amodel', 'tracking', 'nuscenes_att', 'velocity']
    if opt.use_lfa:
      loss_states = ['pc_lfa_feat']
    else:
      loss_states = []
    loss_states = ['tot'] + loss_states + [k for k in loss_order if k in opt.heads]
    
    loss = GenericLoss(opt)
    return loss_states, loss


  def debug(self, batch, output, iter_id, dataset):
    opt = self.opt
    if 'pre_hm' in batch:
      output.update({'pre_hm': batch['pre_hm']})
    dets = fusion_decode(output, K=opt.K, opt=opt) # compute 2D box, store outputs and overwrite primary with secondary head
    
    for k in dets:
      dets[k] = dets[k].detach().cpu().numpy()
    dets_gt = batch['meta'][0]['gt_det']
    for i in range(1):
      debugger = Debugger(opt=opt, dataset=dataset)
      img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm', trans=self.opt.hm_transparency)
      debugger.add_blend_img(img, gt, 'gt_hm', trans=self.opt.hm_transparency)
      
      debugger.add_img(img, img_id='img')
      
      # show point clouds
      if opt.pointcloud:
        pc_2d = batch['pc_2d'][i].detach().cpu().numpy()
        pc_3d = None
        pc_N = batch['pc_N'][i].detach().cpu().numpy()
        debugger.add_img(img, img_id='pc')
        debugger.add_pointcloud(pc_2d, pc_N, img_id='pc')
        
        if 'pc_hm' in opt.pc_feat_lvl:
          channel = opt.pc_feat_channels['pc_hm']
          pc_hm = debugger.gen_colormap(batch['pc_hm'][i][channel].unsqueeze(0).detach().cpu().numpy())
          debugger.add_blend_img(img, pc_hm, 'pc_hm', trans=self.opt.hm_transparency)
        
        if 'pc_dep' in opt.pc_feat_lvl:
          channel = opt.pc_feat_channels['pc_dep']
          pc_box_hm = batch['pc_box_hm'][i][channel].unsqueeze(0).detach().cpu().numpy()
          debugger.add_overlay_img(img, pc_box_hm, 'pc_box_hm_gt')

      if 'pre_img' in batch:
        pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img_pred')
        debugger.add_img(pre_img, 'pre_img_gt')
        if 'pre_hm' in batch:
          pre_hm = debugger.gen_colormap(
            batch['pre_hm'][i].detach().cpu().numpy())
          debugger.add_blend_img(pre_img, pre_hm, 'pre_hm', trans=self.opt.hm_transparency)

      debugger.add_img(img, img_id='out_pred')
      if 'ltrb_amodel' in opt.heads:
        debugger.add_img(img, img_id='out_pred_amodel')
        debugger.add_img(img, img_id='out_gt_amodel')

      # Predictions
      for k in range(len(dets['scores'][i])):
        if dets['scores'][i, k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
            dets['scores'][i, k], img_id='out_pred')

          if 'ltrb_amodel' in opt.heads:
            debugger.add_coco_bbox(
              dets['bboxes_amodel'][i, k] * opt.down_ratio, dets['clses'][i, k],
              dets['scores'][i, k], img_id='out_pred_amodel')

          if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
            debugger.add_coco_hp(
              dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')
          
        if opt.pointcloud and 'pc_dep' in opt.pc_feat_lvl:
          channel = opt.pc_feat_channels['pc_dep']
          pc_box_hm = output['pc_box_hm'][i][channel].unsqueeze(0).detach().cpu().numpy()
          debugger.add_overlay_img(img, pc_box_hm, 'pc_box_hm_pred')
            
      # Ground truth
      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt['scores'][i])):
        if dets_gt['scores'][i][k] > opt.vis_thresh:
          if 'dep' in dets_gt.keys():
            dist = dets_gt['dep'][i][k]
            if len(dist)>1:
              dist = dist[0]
          else:
            dist = -1
          debugger.add_coco_bbox(
            dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
            dets_gt['scores'][i][k], img_id='out_gt', dist=dist)

          if 'ltrb_amodel' in opt.heads:
            debugger.add_coco_bbox(
              dets_gt['bboxes_amodel'][i, k] * opt.down_ratio, 
              dets_gt['clses'][i, k],
              dets_gt['scores'][i, k], img_id='out_gt_amodel')

          if 'hps' in opt.heads and \
            (int(dets['clses'][i, k]) == 0):
            debugger.add_coco_hp(
              dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

      if 'hm_hp' in opt.heads:
        pred = debugger.gen_colormap_hp(
          output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp', trans=self.opt.hm_transparency)
        debugger.add_blend_img(img, gt, 'gt_hmhp', trans=self.opt.hm_transparency)


      if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
        dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
        calib = batch['meta'][0]['calib'].detach().numpy() \
                if 'calib' in batch['meta'][0] else None
        det_pred = generic_post_process(opt, dets, 
          batch['meta'][0]['c'].cpu().numpy(), batch['meta'][0]['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)
        det_gt = generic_post_process(opt, dets_gt, 
          batch['meta'][0]['c'].cpu().numpy(), batch['meta'][0]['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib, is_gt=True)

        # Plot 3D bounding boxes projected in camera frame
        debugger.add_3d_detection(
          batch['meta'][0]['img_path'][i], batch['meta'][0]['flipped'][i],
          det_pred[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_pred')
        debugger.add_3d_detection(
          batch['meta'][0]['img_path'][i], batch['meta'][0]['flipped'][i], 
          det_gt[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_gt')
        
        pc_3d = None
        if opt.pointcloud:
          pc_3d=copy.copy(batch['pc_3d'].cpu().numpy())

        # Plot 3D bounding boxes projected in BEV (with velocity vector)
        debugger.add_bird_views(det_pred[i], det_gt[i], vis_thresh=opt.vis_thresh, 
          img_id='bird_pred_gt', pc_3d=pc_3d, show_velocity=opt.show_velocity)
        debugger.add_bird_views([], det_gt[i], vis_thresh=opt.vis_thresh, 
          img_id='bird_gt', pc_3d=pc_3d, show_velocity=opt.show_velocity)

      # Early Fusion radar map
      if opt.use_early_fusion:
        debugger.plot_radar2img('PC Early Fusion', img, batch)

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader, eval_frustum=None):
    return self.run_epoch('train', epoch, data_loader, eval_frustum=eval_frustum) # for train return empty dict
