from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
from progress.bar import Bar
import time
import torch
import math

from model.model import create_model, load_model
from model.decode import fusion_decode #, generic_decode
from model.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform, affine_transform, transform_preds
from utils.image import draw_umich_gaussian, gaussian_radius
from utils.post_process import generic_post_process
from utils.debugger import Debugger
from utils.tracker import Tracker
from dataset.dataset_factory import get_dataset

class Detector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(
      opt.arch, opt.heads, opt.head_conv, opt=opt)
    self.model = load_model(self.model, opt.load_model, opt)
    self.model = self.model.to(opt.device)
    # Put model in eval mode
    self.model.eval()

    self.opt = opt
    self.trained_dataset = get_dataset(opt.dataset)
    self.mean = np.array(
      self.trained_dataset.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(
      self.trained_dataset.std, dtype=np.float32).reshape(1, 1, 3)
    self.pause = not opt.no_pause
    self.rest_focal_length = self.trained_dataset.rest_focal_length \
      if self.opt.test_focal_length < 0 else self.opt.test_focal_length
    self.flip_idx = self.trained_dataset.flip_idx
    self.cnt = 0
    self.pre_images = None
    self.pre_image_ori = None
    self.tracker = Tracker(opt)
    self.debugger = Debugger(opt=opt, dataset=self.trained_dataset)

  def flip_calib(self, meta) -> np.ndarray:
    """
    "Flip" calibration matrix.

    :meta Meta data
    """
    # Get calib 
    calib = meta['calib'].copy()
    width = meta['width']

    # Get principal point x coordinate
    px = calib[0,2]
    # "Flip" / Transform principal point
    calib[0,2] = width-px

    return calib

  def run(self, image_or_path_or_tensor, meta={}):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, track_time, tot_time, display_time = 0, 0, 0, 0
    self.debugger.clear()
    start_time = time.time()
    pre_processed = False
    pc_hm = None

    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pc_hm = image_or_path_or_tensor.get('pc_hm', None)
      pc_hm_add = image_or_path_or_tensor.get('pc_hm_add')
      trans_original = image_or_path_or_tensor.get('trans_original')
      pc_ef = image_or_path_or_tensor.get('pc_ef')
      pc_snap_proj = image_or_path_or_tensor.get('pc_snap_proj')
      pc_3d = image_or_path_or_tensor.get('pc_3d')
      pc_N = image_or_path_or_tensor.get('pc_N') # number of radar points per image
      


      if pc_hm is not None:
        if self.opt.flip_test:
          # Flip image-like maps themselves
          flipped = torch.flip(pc_hm, [3])
          flipped_add = torch.flip(pc_hm_add, [3])
          flipped_pc_ef = torch.flip(pc_ef, [3])
          flipped_pc_snap_proj = torch.flip(pc_snap_proj, [3])
          # Flip the values of all variables in x-axis
          channel = self.opt.pc_feat_channels['pc_vx']
          flipped[0, channel, :, :] *= -1
          flipped_add[0, 1, :, :] *= -1 # posx at pos 1
          channel_vx = self.opt.early_fusion_channels.index('vx_comp')
          flipped_pc_ef[0, channel_vx, :,:] *= -1
          # Concatenate to unflipped versions
          pc_hm = torch.cat((pc_hm, flipped), axis=0)
          pc_hm_add = torch.cat((pc_hm_add, flipped_add), axis=0)
          pc_ef = torch.cat((pc_ef, flipped_pc_ef), axis=0)
          pc_snap_proj = torch.cat((pc_snap_proj, flipped_pc_snap_proj), axis=0)
          # Add trans_original second time
          trans_original = torch.cat((trans_original, trans_original), axis=0)


          # Flip 3d pc
          pc_3d_flipped = copy.deepcopy(pc_3d)
          pc_3d_flipped[0, 0,:] *= -1  # flip the x position (in camera CS)
          pc_3d_flipped[0, 8,:] *= -1  # flip the x_comp velocity (in camera CS)
          pc_3d = torch.cat((pc_3d, pc_3d_flipped), axis=0)

          pc_N = torch.cat((pc_N, pc_N), axis=0)

        pc_hm = pc_hm.to(self.opt.device, non_blocking=self.opt.non_block_test)
        pc_hm_add = pc_hm_add.to(self.opt.device, non_blocking=self.opt.non_block_test)
        pc_ef = pc_ef.to(self.opt.device, non_blocking=self.opt.non_block_test)
        pc_snap_proj = pc_snap_proj.to(self.opt.device, non_blocking=self.opt.non_block_test)
        pc_3d = pc_3d.to(self.opt.device, non_blocking=self.opt.non_block_test)
        trans_original = trans_original.to(device=self.opt.device, non_blocking=self.opt.non_block_test, dtype=torch.float32)
      pre_processed = True

    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.opt.test_scales:
      scale_start_time = time.time()
      if not pre_processed:
        # not prefetch testing
        images, meta = self.pre_process(image, scale, meta)
      else:
        # prefetch testing
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        if 'pre_dets' in pre_processed_images['meta']:
          meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
        if 'cur_dets' in pre_processed_images['meta']:
          meta['cur_dets'] = pre_processed_images['meta']['cur_dets']

      images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)
      pre_hms, pre_inds = None, None
      if self.opt.tracking:
        if self.pre_images is None:
          print('Initialize tracking!')
          self.pre_images = images
          self.tracker.init_track(meta['pre_dets'])
        if self.opt.pre_hm:
          pre_hms, pre_inds = self._get_additional_inputs(
            self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
      
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      # Create artificial batch to process
      batch = {}
      batch['image'] = images
      batch['pc_hm'] = pc_hm
      batch['pc_hm_add'] = pc_hm_add

      calib = torch.from_numpy(meta['calib']).float().to(images.device).squeeze(0)

      if self.opt.flip_test:
        # Add calib twice
        # calib_flipped = self.flip_calib(meta)
        # calib_flipped = torch.from_numpy(calib_flipped).float().to(images.device).squeeze(0)
        
        # Actually, this is not 100% correct for a flipped image.
        # However, when flipping, we have to get the hm data etc. differently as well and not just flip it.
        # Therefore, we just ignore the incorrectness here.
        calib_flipped = calib.clone()

        calib = torch.cat((calib.unsqueeze(0), calib_flipped.unsqueeze(0)), axis=0)

        meta_flipped = copy.deepcopy(meta)
        meta_flipped['calib'] = calib_flipped

        meta = [meta, meta_flipped]
      
      else: 
        calib = calib.unsqueeze(0)
        meta = [meta]

      # Convert elements in meta to tenors
      for i in range(len(meta)):
        for _, elem in enumerate(meta[i]):
          if type(meta[i][elem]) == torch.Tensor:
            meta[i][elem].to(device=images.device)
          else:
            temp = torch.tensor(meta[i][elem], device=images.device)
          if temp.shape.__len__() == 0:
            temp = temp.unsqueeze(0)
          meta[i][elem] = temp

      batch['calib'] = calib
      batch['trans_original'] = trans_original
      batch['meta'] = meta

      if self.opt.use_early_fusion:
        batch['pc_ef'] = pc_ef
      if self.opt.use_lfa:
        batch['pc_snap_proj'] = pc_snap_proj
        batch['pc_3d'] = pc_3d
        batch['pc_N'] = pc_N

      output, dets, forward_time = self.process(
        batch, pre_inds, return_time=True)

      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      dets = self.post_process(dets, meta, scale)
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)

      if self.opt.debug >= 2:
        self.debug(
          self.debugger, images, dets, output, scale, 
          pre_images=self.pre_images if not self.opt.no_pre_img else None, 
          pre_hms=pre_hms)

    results = self.merge_outputs(detections)  # filter out detections with low confidence score
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    
    if self.opt.tracking:
      public_det = meta['cur_dets'] if self.opt.public_det else None
      results = self.tracker.step(results, public_det)
      self.pre_images = images

    tracking_time = time.time()
    track_time += tracking_time - end_time
    tot_time += tracking_time - start_time

    if self.opt.debug >= 1:
      if self.opt.pointcloud and self.opt.use_sec_heads and len(self.opt.test_scales) == 1:
        self.show_results(self.debugger, image, results, output=output)
      else:
        self.show_results(self.debugger, image, results)
    self.cnt += 1

    show_results_time = time.time()
    display_time += show_results_time - end_time
    
    ret = {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time, 'track': track_time,
            'display': display_time}
    if self.opt.save_video:
      try:
        ret.update({'generic': self.debugger.imgs['generic']})
      except:
        pass
    return ret


  def _transform_scale(self, image, scale=1):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_short > 0: # ?
      if height < width:
        inp_height = self.opt.fix_short
        inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
      else:
        inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
        inp_width = self.opt.fix_short
      c = np.array([width / 2, height / 2], dtype=np.float32)
      s = np.array([width, height], dtype=np.float32)
    elif self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, c, s, inp_width, inp_height, height, width


  def pre_process(self, image, scale, input_meta={}):
    n_channels = 3

    resized_image, c, s, inp_width, inp_height, height, width = \
      self._transform_scale(image)
    # Rotation set to 0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    out_height =  inp_height // self.opt.down_ratio
    out_width =  inp_width // self.opt.down_ratio
    trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    # Normalize rgb channels of input image
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)


    images = inp_image.transpose(2, 0, 1).reshape(1, n_channels, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
             if 'calib' in input_meta else \
             self._get_default_calib(width, height)}
    meta.update({'c': c, 's': s, 'height': height, 'width': width,
            'out_height': out_height, 'out_width': out_width,
            'inp_height': inp_height, 'inp_width': inp_width,
            'trans_input': trans_input, 'trans_output': trans_output})
    if 'pre_dets' in input_meta:
      meta['pre_dets'] = input_meta['pre_dets']
    if 'cur_dets' in input_meta:
      meta['cur_dets'] = input_meta['cur_dets']
    return images, meta


  def _trans_bbox(self, bbox, trans, width, height):
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox


  def _get_additional_inputs(self, dets, meta, with_hm=True):
    trans_input, trans_output = meta['trans_input'], meta['trans_output']
    inp_width, inp_height = meta['inp_width'], meta['inp_height']
    out_width, out_height = meta['out_width'], meta['out_height']
    input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

    output_inds = []
    for det in dets:
      if det['score'] < self.opt.pre_thresh:
        continue
      bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
      bbox_out = self._trans_bbox(
        det['bbox'], trans_output, out_width, out_height)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if with_hm:
          draw_umich_gaussian(input_hm[0], ct_int, radius)
        ct_out = np.array(
          [(bbox_out[0] + bbox_out[2]) / 2, 
           (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
        output_inds.append(ct_out[1] * out_width + ct_out[0])
    if with_hm:
      input_hm = input_hm[np.newaxis]
      if self.opt.flip_test:
        input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
      input_hm = torch.from_numpy(input_hm).to(self.opt.device)
    output_inds = np.array(output_inds, np.int64).reshape(1, -1)
    output_inds = torch.from_numpy(output_inds).to(self.opt.device)
    return input_hm, output_inds


  def _get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = output['hm'].sigmoid_()
    if 'hm_hp' in output:
      output['hm_hp'] = output['hm_hp'].sigmoid_()
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      output['dep'] *= self.opt.depth_scale
    if 'dep_sec' in output and self.opt.sigmoid_dep_sec:
      output['dep_sec'] = 1. / (output['dep_sec'].sigmoid() + 1e-6) - 1.
    return output


  def _flip_output(self, output):
    average_flips = ['hm', 'wh', 'dep', 'dim', 'dep_sec']
    neg_average_flips = ['amodel_offset']
    single_flips = ['ltrb', 'nuscenes_att', \
                    # 'velocity', \
                    'ltrb_amodel', 'reg', \
                    'hp_offset', 'rot', 'tracking', 'pre_hm', 'rot_sec', 'pc_lfa_feat']
    for head in output:
      if head in average_flips:
        # Take average of unflipped and flipped inputs
        # Average is calculated by flipping the flipped outputs back
        output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
      if head in neg_average_flips:
        flipped_tensor = flip_tensor(output[head][1:2])
        # Only unflip parts of the tensor.
        flipped_tensor[:, 0::2] *= -1
        output[head] = (output[head][0:1] + flipped_tensor) / 2
      if head in single_flips:
        # Only use the outputs of the unflipped image
        output[head] = output[head][0:1]
      if head == 'hps':
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
      if head == 'hm_hp':
        output['hm_hp'] = (output['hm_hp'][0:1] + \
          flip_lr(output['hm_hp'][1:2], self.flip_idx)) / 2
      if head == 'velocity':
        flipped_tensor = flip_tensor(output[head].clone()[1:2])
        flipped_tensor[:,0,:,:] *= -1
        output[head] = (output[head][0:1] + flipped_tensor) / 2

    return output


  def process(self, batch, pre_inds=None, return_time=False):
    images = batch['image']
    with torch.no_grad():

      torch.cuda.synchronize()
      output = self.model(batch)[-1]
      output = self._sigmoid_output(output)
      output.update({'pre_inds': pre_inds})
      if self.opt.flip_test:
        output = self._flip_output(output)
      torch.cuda.synchronize()
      forward_time = time.time()
      
      dets = fusion_decode(output, K=self.opt.K, opt=self.opt) # 3D box encoder to put together predictions

      torch.cuda.synchronize()
      for k in dets:
        dets[k] = dets[k].detach().cpu().numpy()
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    meta = meta[0]
    c = meta['c'].cpu().numpy()
    s = meta['s'].cpu().numpy()
    out_height = meta['out_height'].cpu().numpy()[0]
    out_width = meta['out_width'].cpu().numpy()[0]
    height = meta['height'].cpu().numpy()[0]
    width = meta['width'].cpu().numpy()[0]
    calib = meta['calib'].cpu().numpy()

    dets = generic_post_process(
      self.opt, dets, [c], [s],
      out_height, out_width, self.opt.num_classes,
      [calib], height, width)

    self.this_calib = calib
    
    if scale != 1:
      for i in range(len(dets[0])):
        for k in ['bbox', 'hps']:
          if k in dets[0][i]:
            dets[0][i][k] = (np.array(
              dets[0][i][k], np.float32) / scale).tolist()
    return dets[0]

  def merge_outputs(self, detections):
    assert len(self.opt.test_scales) == 1, 'multi_scale not supported!'
    results = []
    counter = 0
    for i in range(len(detections[0])):
      det = detections[0][i]
      # filter out detections with low score and negative dimensions
      if det['score'] > self.opt.out_thresh and all(det['dim'] > 0):
        results.append(det)
    return results

  def debug(self, debugger, images, dets, output, scale=1, 
    pre_images=None, pre_hms=None):
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm', trans=self.opt.hm_transparency)
    if 'hm_hp' in output:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp', trans=self.opt.hm_transparency)

    if pre_images is not None:
      pre_img = pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
      pre_img = np.clip(((
        pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
      debugger.add_img(pre_img, 'pre_img')
      if pre_hms is not None:
        pre_hm = debugger.gen_colormap(
          pre_hms[0].detach().cpu().numpy())
        debugger.add_blend_img(pre_img, pre_hm, 'pre_hm', trans=self.opt.hm_transparency)


  def show_results(self, debugger, image, results, output=None):
    debugger.add_img(image, img_id='generic')
    if self.opt.tracking:
      debugger.add_img(self.pre_image_ori if self.pre_image_ori is not None else image, 
        img_id='previous')
      self.pre_image_ori = image
    
    for j in range(len(results)):
      if results[j]['score'] > self.opt.vis_thresh:
        item = results[j]
        if ('bbox' in item):
          sc = item['score'] if self.opt.demo == '' or \
            not ('tracking_id' in item) else item['tracking_id']
          sc = item['tracking_id'] if self.opt.show_track_color else sc
          
          debugger.add_coco_bbox(
            item['bbox'], item['class'] - 1, sc, img_id='generic')

        if 'tracking' in item:
          debugger.add_arrow(item['ct'], item['tracking'], img_id='generic')
        
        tracking_id = item['tracking_id'] if 'tracking_id' in item else -1
        if 'tracking_id' in item and self.opt.demo == '' and \
          not self.opt.show_track_color:
          debugger.add_tracking_id(
            item['ct'], item['tracking_id'], img_id='generic')

        if (item['class'] in [1, 2]) and 'hps' in item:
          debugger.add_coco_hp(item['hps'], tracking_id=tracking_id,
            img_id='generic')

    if len(results) > 0 and \
      'dep' in results[0] and 'alpha' in results[0] and 'dim' in results[0]:
      debugger.add_3d_detection(
        image if not self.opt.qualitative else cv2.resize(
          debugger.imgs['pred_hm'], (image.shape[1], image.shape[0])), 
        False, results, self.this_calib,
        vis_thresh=self.opt.vis_thresh, img_id='ddd_pred')
      debugger.add_bird_view(
        results, vis_thresh=self.opt.vis_thresh,
        img_id='bird_pred', cnt=self.cnt)
      if self.opt.show_track_color and self.opt.debug == 4:
        del debugger.imgs['generic'], debugger.imgs['bird_pred']

      # Plot pred box hm
      if output != None and self.opt.pointcloud and 'pc_dep' in self.opt.pc_feat_lvl:
        channel = self.opt.pc_feat_channels['pc_dep']
        pc_box_hm = output['pc_box_hm'][0][channel].unsqueeze(0).detach().cpu().numpy()
        debugger.add_overlay_img(image, pc_box_hm, 'pc_box_hm_pred')
        if self.opt.flip_test:
          flipped_image = image[:, :, :, ::-1]
          debugger.add_img(flipped_image, img_id='generic_flipped')
          pc_box_hm = output['pc_box_hm'][1][channel].unsqueeze(0).detach().cpu().numpy()
          debugger.add_overlay_img(flipped_image, pc_box_hm, 'pc_box_hm_pred_flip')
      
    if 'ddd_pred' in debugger.imgs:
      debugger.imgs['generic'] = debugger.imgs['ddd_pred']
    if self.opt.debug == 4:
      debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt))
    else:
      debugger.show_all_imgs(pause=self.pause)
  

  def reset_tracking(self):
    self.tracker.reset()
    self.pre_images = None
    self.pre_image_ori = None
