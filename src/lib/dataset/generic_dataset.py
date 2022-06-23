from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import cv2
import os
from collections import defaultdict
import matplotlib.pyplot as plt

import pycocotools.coco as coco
import torch.utils.data as data
from model.networks.lfanet import set_pc_box_hm


from utils.image import color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
from utils.pointcloud import get_dist_thresh_nabati, map_pointcloud_to_image, pc_hm_to_box, get_dist_thresh
import copy
from utils.ddd_utils import compute_box_3d, project_to_image, draw_box_3d, vgt_to_vrad
from utils.image import transform_preds_with_trans
from utils.snapshot import generate_snap_BEV, generate_snap_proj
from utils.eval_frustum import EvalFrustum


class GenericDataset(data.Dataset):
  default_resolution = None
  num_categories = None
  class_name = None
  # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
  # Not using 0 because 0 is used for don't care region and ignore loss.
  cat_ids = None
  max_objs = None
  rest_focal_length = 1200
  num_joints = 17
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
           [4, 6], [3, 5], [5, 6], 
           [5, 7], [7, 9], [6, 8], [8, 10], 
           [6, 12], [5, 11], [11, 12], 
           [12, 14], [14, 16], [11, 13], [13, 15]]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                      dtype=np.float32)
  _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
  ignore_val = 1
  nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4], 
    4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7], 7: [5, 6, 7]}
  
  pc_mean = np.zeros((18,1))
  pc_std = np.ones((18,1))
  img_ind = 0


  def __init__(self, opt=None, split=None, ann_path=None, img_dir=None, eval_frustum:EvalFrustum = None):
    super(GenericDataset, self).__init__()
    if opt is not None and split is not None:
      self.split = split
      self.opt = opt
      self._data_rng = np.random.RandomState(123)
      self.enable_meta = True if (opt.run_dataset_eval and split in \
          ["val", "mini_val", "tiny_val", "wee_val", "nano_val", "debug_val", "test", 'night_and_rain_val', 'night_rain_val','night_val']) or opt.eval else False
    
    self.eval_frustum = eval_frustum
    
    # Random state for blackin    
    if 'val' in split:
      # Random seed for comparability in validation
      self.blackin_rs = np.random.RandomState(1)
    else:
      # 'Real' randomness for training
      self.blackin_rs = np.random.RandomState()


    if ann_path is not None and img_dir is not None:
      print('==> initializing {} data from {}, \n images from {} ...'.format(
        split, ann_path, img_dir))
      self.coco = coco.COCO(ann_path) # create coco object to handle annotations
      self.images = self.coco.getImgIds()

      if opt.tracking:
        if not ('videos' in self.coco.dataset):
          self.fake_video_data()
        print('Creating video index!')
        self.video_to_images = defaultdict(list)
        for image in self.coco.dataset['images']:
          self.video_to_images[image['video_id']].append(image)
      
      self.img_dir = img_dir


  def __getitem__(self, index):
    # This function is called when a sample of the dataset is used as input to network.
    # It is consecutively applied to all samples of a batch. The batch as a whole the is forwarded
    # into the network.
    
    opt = self.opt
    img, anns, img_info, img_path = self._load_data(index)

    # May drop image if self.opt.blackin > 0
    # Likeliness of blackin is specified in opts,
    # Validation uses random seed, training does not
    blackin = self.opt.blackin
    if blackin > 0.0 and self.blackin_rs.choice(2, p=[1-blackin, blackin]) == 1:
      img = np.zeros_like(img)
    
    height, width = img.shape[0], img.shape[1]

    ## Get center and scale from image
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
      else np.array([img.shape[1], img.shape[0]], np.float32)
    aug_s, rot, flipped = 1, 0, 0

    calib = self._get_calib(img_info, width, height)

    # # Transformation to original image size
    # trans_original = get_affine_transform(
    #   c, s, rot, [opt.output_w, opt.output_h], inv=1)

    ## data augmentation for training set
    if 'train' in self.split:
      c, aug_s, rot = self._get_aug_param(c, s, width, height)
      s = s * aug_s
      if np.random.random() < opt.flip:
        flipped = 1

        img = img[:, ::-1, :]
        img = img.copy()

        anns = self._flip_anns(anns, width, img_info['camera_intrinsic'])
        

    # Transformation to input image size [default 800x448]
    trans_input = get_affine_transform(
      c, s, rot, [opt.input_w, opt.input_h])
    # Transformation to output image size [default 200x112]
    trans_output = get_affine_transform(
      c, s, rot, [opt.output_w, opt.output_h])
    # Transformation to original image size while regarding the augmentation
    trans_original = get_affine_transform(
      c, s, rot, [opt.output_w, opt.output_h], inv=1).astype(np.float32)

    if flipped and (self.opt.use_early_fusion or self.opt.snap_method=='proj'):
      # Calculate transformation for flipped image
      c_f = c.copy()
      c_f[0] = width - c[0]
      s_f = s.copy()
      trans_input_flipped = get_affine_transform(
                      c_f, s_f, rot, [opt.input_w, opt.input_h])
      trans_output_flipped = get_affine_transform(
                c_f, s_f, rot, [opt.output_w, opt.output_h])
      # trans_orig_flipped = get_affine_transform(
      #           c_f, s_f, rot, [width, height])

    inp = self._get_input(img, trans_input)
    # cv2.imshow('inp', inp)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    ret = {'image': inp} # store everything that should be inside a sample
    gt_det = {'scores': [], 'clses': [], 'cts': [], 'bboxes': [], 'bboxes_amodel': []}

    ret['trans_original'] = trans_original # Save this to later use it for dist thresh etc.


    # print(f'c: {c} | s: {s}')

    #  load point cloud data
    if opt.pointcloud:
      
      pc_hm, pc_hm_add, pc_N, pc_2d, pc_3d, pc_ef, pc_snap_proj  = self._load_pc_data(img, img_info, 
        trans_input, trans_output, flipped)

      if 'train' in self.split:
        # Get augmentation for EF training
        # trans_ef = get_affine_transform(c, s, rot, [pc_ef.shape[2], pc_ef.shape[1]])
        if self.opt.use_early_fusion:
          # Flip pc_ef if necessary
          if flipped == 1:
            # Apply flipped transformation
            pc_ef = self._get_input_ef(pc_ef, trans_input_flipped)
            pc_ef = np.flip(pc_ef,2).copy()
          else:
            # Apply regular transformation
            pc_ef = self._get_input_ef(pc_ef, trans_input)

        if self.opt.snap_method == 'proj':
          # Flip pc_snap_proj if necessary
          if flipped == 1:
            # Apply flipped transformation
            # NO AUGMENTATION SUPPORTED FOR SNAP GENERATION !!!
            pc_snap_proj = np.flip(pc_snap_proj,2).copy()
      else:
        # Validation, no data augmentation on EF inputs
        if self.opt.use_early_fusion:
          pc_ef = self._get_input_ef(pc_ef, trans_input)

      ret.update({'pc_hm': pc_hm,
                  'pc_hm_add': pc_hm_add,
                  'pc_N': pc_N,
                  'pc_2d': pc_2d,
                  'pc_3d': pc_3d,
                 })
      if self.opt.use_early_fusion:
        ret['pc_ef'] = pc_ef
      if self.opt.snap_method == 'proj':
        ret['pc_snap_proj'] = pc_snap_proj


    pre_cts, track_ids = None, None
    if opt.tracking: # TODO: a lot of changes are not implemented here, since we don't use tracking in the thesis
      pre_image, pre_anns, frame_dist, pre_img_info = self._load_pre_data(
        img_info['video_id'], img_info['frame_id'], 
        img_info['sensor_id'] if 'sensor_id' in img_info else 1)
      if flipped:
        pre_image = pre_image[:, ::-1, :].copy()
        pre_anns = self._flip_anns(pre_anns, width)
        if pc_2d is not None:
          pc_2d = self._flip_pc(pc_2d,  width)
      if opt.same_aug_pre and frame_dist != 0:
        trans_input_pre = trans_input.copy()
        trans_output_pre = trans_output
      else:
        c_pre, aug_s_pre, _ = self._get_aug_param(
          c, s, width, height, disturb=True)
        s_pre = s * aug_s_pre
        trans_input_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.input_w, opt.input_h])
        trans_output_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.output_w, opt.output_h])
      pre_img = self._get_input(pre_image, trans_input_pre)
      pre_hm, pre_cts, track_ids = self._get_pre_dets(
        pre_anns, trans_input_pre, trans_output_pre)
      ret['pre_img'] = pre_img
      if opt.pre_hm:
        ret['pre_hm'] = pre_hm
      if opt.pointcloud:
        pre_pc_hm, pre_pc_hm_add, pre_pc_N, pre_pc_2d, pre_pc_3d, pre_pc_ef = \
        self._load_pc_data(pre_img, pre_img_info, trans_input_pre, 
                           trans_output_pre, flipped)
        ret['pre_pc_hm'] = pre_pc_hm
        ret['pre_pc_hm_add'] = pre_pc_hm_add
        ret['pre_pc_N'] = pre_pc_N
        ret['pre_pc_2d'] = pre_pc_2d
        ret['pre_pc_3d'] = pre_pc_3d
        
    ### init samples
    self._init_ret(ret, gt_det)

    # get velocity transformation matrix
    if "velocity_trans_matrix" in img_info:
      velocity_mat = np.array(img_info['velocity_trans_matrix'], dtype=np.float32)
    else:
      velocity_mat = np.eye(4)
    
    num_objs = min(len(anns), self.max_objs)
    for k in range(num_objs):
      ann = anns[k]
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -999:
        continue

      bbox, bbox_amodel = self._get_bbox_output(
            ann['bbox'], trans_output, height, width)

      if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
        self._mask_ignore_or_crowd(ret, cls_id, bbox)
        continue
      self._add_instance(
        ret, gt_det, k, cls_id, bbox, bbox_amodel, ann, trans_output, aug_s, 
        calib, pre_cts, track_ids)

    if self.opt.debug > 0 or self.enable_meta:
      gt_det = self._format_gt_det(gt_det)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
              'img_path': img_path, 'calib': calib,
              'img_width': img_info['width'], 'img_height': img_info['height'],
              'flipped': flipped, 'velocity_mat':velocity_mat}
      ret['meta'] = [meta] # do not remove []
    ret['calib'] = calib

    # Get calibration matrix for early fusion, scaled to work on input sized images
    # ret['calib_input'] = self._scale_calib(calib, width, height, opt.input_w, opt.input_h)

    return ret


  def get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib

  def _load_image_anns(self, img_id, coco, img_dir):
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

  def _load_data(self, index):
    coco = self.coco
    img_dir = self.img_dir
    img_id = self.images[index]
    img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

    return img, anns, img_info, img_path


  def _load_pre_data(self, video_id, frame_id, sensor_id=1):
    img_infos = self.video_to_images[video_id]
    # If training, random sample nearby frames as the "previous" frame
    # If testing, get the exact prevous frame
    if 'train' in self.split:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
          if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist and \
          (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    else:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == -1 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
      if len(img_ids) == 0:
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == 0 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    rand_id = np.random.choice(len(img_ids))
    img_id, pre_frame_id = img_ids[rand_id]
    frame_dist = abs(frame_id - pre_frame_id)
    img, anns, img_info, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
    return img, anns, frame_dist, img_info


  def _get_pre_dets(self, anns, trans_input, trans_output):
    hm_h, hm_w = self.opt.input_h, self.opt.input_w
    down_ratio = self.opt.down_ratio
    trans = trans_input.copy()
    reutrn_hm = self.opt.pre_hm
    pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
    pre_cts, track_ids = [], []
    for ann in anns:
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -99 or \
         ('iscrowd' in ann and ann['iscrowd'] > 0):
        continue
      bbox = self._coco_box_to_bbox(ann['bbox'])
      bbox[:2] = affine_transform(bbox[:2], trans)
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      max_rad = 1
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius)) 
        max_rad = max(max_rad, radius)
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct0 = ct.copy()
        conf = 1

        ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
        ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
        conf = 1 if np.random.random() > self.opt.lost_disturb else 0
        
        ct_int = ct.astype(np.int32)
        if conf == 0:
          pre_cts.append(ct / down_ratio)
        else:
          pre_cts.append(ct0 / down_ratio)

        track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
        if reutrn_hm:
          draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

        if np.random.random() < self.opt.fp_disturb and reutrn_hm:
          ct2 = ct0.copy()
          # Hard code heatmap disturb ratio, haven't tried other numbers.
          ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
          ct2[1] = ct2[1] + np.random.randn() * 0.05 * h 
          ct2_int = ct2.astype(np.int32)
          draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

    return pre_hm, pre_cts, track_ids

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


  def _get_aug_param(self, c, s, width, height, disturb=False):
    """
    :param c: center of image
    :param s: maximum of image size (e.g. 1600 for nuScenes)
    """
    if (self.opt.rand_crop) and not disturb:
      # Perform random cropping
      aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
      w_border = self._get_border(128, width)
      h_border = self._get_border(128, height)
      c[0] = np.random.randint(low=w_border, high=width - w_border)
      c[1] = np.random.randint(low=h_border, high=height - h_border)
    else:
      sf = self.opt.scale
      cf = self.opt.shift
      # if type(s) == float:
      #   s = [s, s]
      temp = np.random.randn()*cf
      c[0] += s * np.clip(temp, -2*cf, 2*cf)
      c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      aug_s = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    
    if np.random.random() < self.opt.aug_rot:
      rf = self.opt.rotate
      rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)
    else:
      rot = 0
    
    return c, aug_s, rot


  def _flip_anns(self, anns, width, camera_intrinistic):
    # Flip: mirror along z-axis 
    # Need to flip:
    #   - location (x-coordinate)
    #   - rotation_y
    #   - amodel_center (x-coordinate)
    #   - velocity
    #   - velocity_cam
    #   - bbox
    #   - alpha

    for k in range(len(anns)):
      bbox = anns[k]['bbox']
      # bbox format (x,y,width,height)
      anns[k]['bbox'] = [
        width - 1  - bbox[0] - bbox[2], bbox[1], bbox[2], bbox[3]]
      
      if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
        keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(
          self.num_joints, 3)
        keypoints[:, 0] = width - keypoints[:, 0] - 1
        for e in self.flip_idx:
          keypoints[e[0]], keypoints[e[1]] = \
            keypoints[e[1]].copy(), keypoints[e[0]].copy()
        anns[k]['keypoints'] = keypoints.reshape(-1).tolist()

      if 'rot' in self.opt.heads and 'alpha' in anns[k]:
        anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 \
                           else - np.pi - anns[k]['alpha']

      if 'amodel_offset' in self.opt.heads and 'amodel_center' in anns[k]:
        anns[k]['amodel_center'][0] = width - 1 - anns[k]['amodel_center'][0]

      if self.opt.velocity and 'velocity' in anns[k]:
        # anns[k]['velocity'] = [-10000, -10000, -10000]
        anns[k]['velocity'][0] *= -1 # mirror over z axis

      # Velocity in camera CS
      if self.opt.velocity and 'velocity_cam' in anns[k]:
        anns[k]['velocity_cam'][0] *= -1 # mirror over z axis (Camera CS)

      # Mirror location of 3D bounding box
      if 'location' in anns[k]:
        anns[k]['location'][0] *= -1 # mirror over z axis (Camera CS)

      if 'rot' in self.opt.heads and 'rotation_y' in anns[k]:
        anns[k]['rotation_y'] = np.pi - anns[k]['rotation_y'] if anns[k]['rotation_y'] > 0 \
                           else - np.pi - anns[k]['rotation_y']

    return anns


  def _load_pc_data(self, img, img_info, inp_trans, out_trans, flipped=0):
    """
    Load the radar pointcloud data and get the information from it in form
    of a heatmap.
    
    :return: pc_z tensor. Is to be understood as all points
                          being projected into the image plane
                          and having their depth (or z-coordinate
                          in camera CS) as a channel information.
    :return: 
     """
    # img_height, img_width = img.shape[0], img.shape[1]
    radar_pc = np.array(img_info.get('radar_pc', None))
    # radar_delta_ts = np.array(img_info.get('radar_delta_ts', None))
    if radar_pc is None:
      return None, None, None, None

    # Get depth to points (z-coordinate in camera CS)
    depth = radar_pc[2,:]
    
    # Filter points by distance
    # This is to limit computational effort and is done
    # since the camera cannot detect over this depth.
    if self.opt.max_pc_depth > 0:
      mask = (depth <= self.opt.max_pc_depth)
      radar_pc = radar_pc[:,mask]
      # radar_delta_ts = radar_delta_ts[:, mask]
      depth = depth[mask]

    # Add an offset to radar points in z axis
    # (z-coordinate in radar CS)
    if self.opt.pc_z_offset != 0:
      radar_pc[1,:] -= self.opt.pc_z_offset
    
    # Map points to the image
    # pc_2d contains 3 coordinates but is to be understood
    # as the points of the pc being projected into the 
    # image plane with a channel information about the
    # depth. The points are not 3D points.
    pc_2d, mask = map_pointcloud_to_image(radar_pc, 
                      np.array(img_info['camera_intrinsic']), 
                      img_shape=(img_info['width'],img_info['height']))

    # Construct 
    
    # Filter all points whether or not they would have 
    # been projected into the image.
    pc_3d = radar_pc[:,mask] 
    # pc_dts = radar_delta_ts[:,mask]

    # Sort points by depth 
    # important for the generation of the heatmap later
    ind = np.argsort(pc_2d[2,:])
    pc_2d = pc_2d[:,ind]
    pc_3d = pc_3d[:,ind]
    # pc_dts = pc_dts[:,ind]


    ### Construct heatmap for pointcloud
    pc_hm, pc_hm_add, pc_N, pc_2d, pc_3d = self._process_pc(
                pc_2d, pc_3d, img, inp_trans, out_trans, img_info, flipped)


    ## DEBUG PLOT
    # Plot the point cloud in 2D for visualization in report
    # fig,ax = plt.subplots()
    # plt.fill(np.array([0,np.arcsin(np.deg2rad(35))*np.amax(pc_3d[2,:])+1, -np.arcsin(np.deg2rad(35))*np.amax(pc_3d[2,:])+1]), np.array([0,np.amax(pc_3d[2,:])+1,np.amax(pc_3d[2,:])+1]),c=np.array([255,238,170,40])/255)
    # plt.scatter(pc_3d[0,:],pc_3d[2,:],c=np.array([[55/255, 86/255, 35/255]]), s=2)    
    # ax.set_axisbelow(True)
    # ax.grid(True)
    # # plt.tight_layout()
    # plt.show()

    # Init array that can be used for early fusion
    num_ef_channels = len(self.opt.early_fusion_channels)
    # pc_ef = np.zeros((num_ef_channels, self.opt.input_h, self.opt.input_w))

    pc_ef = np.zeros((num_ef_channels, img_info['height'], img_info['width']))
    pc_snap_proj = np.ones((1, img_info['height'], img_info['width']))*-1
    
    if self.opt.use_early_fusion or self.opt.snap_method=='proj':
      # Map points to image and fill pc_ef
      pc_3d_copy = pc_3d.copy() # copy needed to not mess up the original pc_3d
      # Init of radar points on the xz-plane in camera coordinates (y=0)
      radar_points_b = np.zeros((3,pc_3d_copy.shape[1]))
      # Init of radar points on the xz-plane in camera coordinates with offset
      # (+opt.early_fusion_projection_height)
      radar_points_u = np.zeros((3,pc_3d_copy.shape[1]))

      # Fill arrays with radar points:
      # Bottom points
      radar_points_b[[0,2],:] = pc_3d_copy[[0,2],:] # x and z values of radar points
      # To represent points on the floor, the radar points have to be transformed according
      # to the translation of the camera with respect to the ego vehicle frame. 
      # The ego vehicle frame represents the midpoint of the rear vehicle axis and is 
      # therefore likely not exactly the ground but the best approximation we have.
      translation_u = img_info['cs_record_trans'][2] 
      radar_points_b[1,:] += -self.opt.early_fusion_projection_height + translation_u # y values of radar points

      # Upper points
      radar_points_u[[0,2],:] = pc_3d_copy[[0,2],:] # x and z values of radar points
      radar_points_u[1,:] += translation_u 

      # If image is flipped, unflip radar points.
      # This has to be done to guarantee correct mapping through the camera intrinistics
      # The camera matrix is given in the unflipped image, when applied to flipped points 
      # this gives incorrect results (mainly due too principal point that should be flipped)
      if flipped == 1:
        radar_points_b[0,:] *= -1
        radar_points_u[0,:] *= -1

      # Transform points into image plane

      calib_input = img_info['calib']

      radar_pixels_b = project_to_image(radar_points_b.T, calib_input).astype(np.int)
      radar_pixels_u = project_to_image(radar_points_u.T, calib_input).astype(np.int)

      radar_pixels_b[:,0] = np.clip(radar_pixels_b[:,0], 0, img_info['width']-1)
      radar_pixels_b[:,1] = np.clip(radar_pixels_b[:,1], 0, img_info['height']-1)
      radar_pixels_u[:,0] = np.clip(radar_pixels_u[:,0], 0, img_info['width']-1)
      radar_pixels_u[:,1] = np.clip(radar_pixels_u[:,1], 0, img_info['height']-1)

      # Fill the blank pc_ef array with the channels required
      for i in np.arange(pc_3d_copy.shape[1]-1,-1,-1): # reverse order to map closer points last

        # Indices for 2D line in image
        pixels_y = np.arange(radar_pixels_b[i,1], radar_pixels_u[i,1]+1)
        
        if self.opt.use_early_fusion:
          # Make pixels wider then just one pixel
          pixels_x_ef = np.arange(np.ceil(np.amax((0,radar_pixels_b[i,0]-self.opt.early_fusion_pixel_width/2+1))), # left boundary
                            np.ceil(np.amin((img_info['width']-1, radar_pixels_b[i,0]+self.opt.early_fusion_pixel_width/2)))+1,dtype=np.int64) # right boundary
          for idx_x in pixels_x_ef:
                      # Assign all channels in one step for Early Fusion
            pc_ef[:, pixels_y, idx_x] = pc_3d_copy[self.opt.early_fusion_channel_indices, i].reshape(num_ef_channels,1)  *  \
                                          np.ones((num_ef_channels, pixels_y.shape[0]))
        if self.opt.snap_method == 'proj':
          # Pixel width of 8 corresponding to 1600->200 (output width)
          # pixels_x_proj = np.arange(np.ceil(np.amax((0,radar_pixels_b[i,0]+1))), # left boundary
          #                   np.ceil(np.amin((img_info['width']-1, radar_pixels_b[i,0])))+1,dtype=np.int64) # right boundary
          
          # for idx_x in pixels_x_proj:
          #   # Only save point index
          pc_snap_proj[0, pixels_y, radar_pixels_b[i,0]] = i
      
      pc_ef = pc_ef.astype(np.float32)
      pc_snap_proj = pc_snap_proj.astype(np.float32)

    # Pad pointclouds with zero / limit in nr of points to avoid size
    # mismatch error in dataloader
    n_points = min(self.opt.max_pc, pc_N)
    pc_2d_ret = np.zeros((pc_2d.shape[0], self.opt.max_pc))
    pc_2d_ret[:, :n_points] = pc_2d[:, :n_points]
    pc_3d_ret = np.zeros((pc_3d.shape[0], self.opt.max_pc))
    pc_3d_ret[:, :n_points] = pc_3d[:, :n_points]
    # pc_dtsz = np.zeros((pc_dts.shape[0], self.opt.max_pc))
    # pc_dtsz[:, :n_points] = pc_dts[:, :n_points]

    return pc_hm, pc_hm_add, pc_N, pc_2d_ret, pc_3d_ret, pc_ef, pc_snap_proj


  def _process_pc(self, pc_2d, pc_3d, img, inp_trans, out_trans, img_info, flipped):  
    """
    Create pillars in 3D and project them to the image. The
    projected pillars are then filled with the features of the pointcloud in
    different channels. 

    :param flipped: True if current image is flipped

    :return: pc_hm np array [out_img_shape,opt.pc_feat_lvl]. 
                    A 2D heatmap (in the plane of the image) with all features
                    of the radar point cloud that are used for the heads.
                    The features are mapped to multiple pixels as projected 
                    pillars.
    :return: pc_hm_add np array [out_img_shape,3] 
                    Additional features of the heatmap that are concatenated
                    later in the frustum association and evalaution of association
                    but they are not supposed to be used as input to the heads. 
                    Thus they are not concatenated here yet.
                    Additional heatmap layers:
                    - [0] delta timestamps between radar sweeps
                    - [1] position in x (camera CS)
                    - [2] position in z (camera CS)
    :return: pc_N int
                    Number of points that lie in downscaled pc_2d
    :return: pc_2d np array
                    Same as input but points masked out that don't lie in
                    downscaled pc_2d
    :return: pc_3d np array
                    Same as input but points masked out that don't lie in
                    downscaled pc_2d


    """  
    # Transform points
    mask = None
    if len(self.opt.pc_feat_lvl) > 0:
      # Transform "image" of pc to output shape and get mask for points
      # that fall outside of output shaped "image".
      pc_feat, mask = self._transform_pc(pc_2d, out_trans, self.opt.output_w, self.opt.output_h)
      # Initialize arrays
      pc_hm = np.zeros((len(self.opt.pc_feat_lvl), self.opt.output_h, self.opt.output_w), np.float32)
      pc_hm_add = np.zeros((2, self.opt.output_h, self.opt.output_w), np.float32)
    
    # Mask out points that don't fit into output shaped "image"
    if mask is not None:
      pc_N = np.array(sum(mask)) # number of points that fit into output "image"
      pc_2d = pc_2d[:,mask]
      pc_3d = pc_3d[:,mask]
    else:
      pc_N = pc_2d.shape[1]

    # Flip points if image is flipped
    if flipped:
      pc_2d = self._flip_pc(pc_2d,  img.shape[1]) # flip pointcloud in image
      pc_3d[0,:] *= -1  # flip the x position (in camera CS)
      pc_3d[8,:] *= -1  # flip the x_comp velocity (in camera CS)


    if self.opt.use_sec_heads and not self.opt.use_lfa: # don't need it in LFANet or EF
      ## "Conventional" approach by Nabati without LFANet
      pc_3d_copy = pc_3d.copy() # copy needed to not mess up the original pc_3d
      # Create pointcloud pillars
      if self.opt.pc_roi_method == "pillars":
        pillar_wh = self.create_pc_pillars(img, img_info, pc_2d, pc_3d_copy, inp_trans, out_trans, flipped)    

      # Generate pointcloud channels for heatmap
      feature_list = copy.copy(self.opt.pc_feat_lvl)
      feature_list.append('pc_dts')
      feature_list.append('pc_pos_x')

      # Loop over points and iterate through pointcloud from farthest to closest
      # since closest should overwrite farther ones in the heatmap. This is 
      # because in the frustum association with the method 'closest' always the
      # closest point is chosen.
      for i in range(pc_N-1, -1, -1):
        for feat in feature_list:
          point = pc_feat[:,i]
          depth = point[2] # z-coordinate (in camera CS)
          ct = np.array([point[0], point[1]]) # x,y-coordinate (in camera CS)
          ct_int = ct.astype(np.int32) # convert to pixel

          if self.opt.pc_roi_method == "pillars":
            wh = pillar_wh[:,i]
            # The pillars are set to end at the height of the center and 
            # start at the full height below that. Change format from w,h to h,w.

            b = [max(ct[1]-wh[1], 0), 
                ct[1], 
                max(ct[0]-wh[0]/2, 0), 
                min(ct[0]+wh[0]/2, self.opt.output_w)]
            b = np.round(b).astype(np.int32) # convert to pixel

            # Add 1 to the upper bounds of the bounding boxes (and subtract 1 from
            # the lower bounds if reaching boundaries). Otherwise, a single line of 
            # pixels will be ignored in the heat maps.
            if b[1] < self.opt.output_h:
              b[1] += 1
            else:
              b[0] += -1
            if b[3] < self.opt.output_w:
              b[3] += 1
            else:
              b[2] += -1
          
          elif self.opt.pc_roi_method == "hm":
            radius = (1.0 / depth) * self.opt.r_a + self.opt.r_b
            radius = gaussian_radius((radius, radius))
            radius = max(0, int(radius))
            x, y = ct_int[0], ct_int[1]
            height, width = pc_hm.shape[1:3]
            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)
            b = np.array([y - top, y + bottom, x - left, x + right])
            b = np.round(b).astype(np.int32)
          
          ## Create heatmap of pointcloud
          # Information might get lost here! The information of the bounding boxes
          # just gets overwritten from next point. The points are sorted by 
          # distance so when using 'closest' as frustum association method 
          # this is not a problem.
          if feat == 'pc_dep':
            channel = self.opt.pc_feat_channels['pc_dep']
            # Assign depth to pillars / hm (default: pillars)
            pc_hm[channel, b[0]:b[1], b[2]:b[3]] = depth
          
          elif feat == 'pc_vx':
            vx = pc_3d_copy[8,i] # vx_comp 
            channel = self.opt.pc_feat_channels['pc_vx']
            # assign vx to pillars / hm (default: pillars)
            pc_hm[channel, b[0]:b[1], b[2]:b[3]] = vx
          
          elif feat == 'pc_vz':
            vz = pc_3d_copy[9,i] # vz_comp
            channel = self.opt.pc_feat_channels['pc_vz']
            # assign vz to pillars / hm (default: pillars)
            pc_hm[channel, b[0]:b[1], b[2]:b[3]] = vz
          
          elif feat == 'pc_rcs':
            rcs = pc_3d_copy[5,i] # rcs
            channel = self.opt.pc_feat_channels['pc_rcs']
            # assign RCS to pillars / hm (default: pillars)
            pc_hm[channel, b[0]:b[1], b[2]:b[3]] = rcs

          elif feat == 'pc_dts':
            # Only add it to added heatmap since it is not supposed to be a 
            # feature to learn by but only used in frustum association
            dts = pc_3d_copy[18][i]
            # assign delta timestamp to pillars / hm (default: pillars)
            pc_hm_add[0, b[0]:b[1], b[2]:b[3]] = dts

          elif feat == 'pc_pos_x':
            # Only add it to added heatmap since it is not supposed to be a 
            # feature to learn by but only used in frustum association
            pos_x = pc_3d_copy[0][i]
            # assign position in x pillars / hm (default: pillars)
            pc_hm_add[1, b[0]:b[1], b[2]:b[3]] = pos_x

    return pc_hm, pc_hm_add, pc_N, pc_2d, pc_3d 


  def create_pc_pillars(self, img, img_info, pc_2d, pc_3d, inp_trans, out_trans, flipped):
    """
    :param flipped: True if current image is flipped
    """

    pillar_wh = np.zeros((2, pc_3d.shape[1])) # [2xN], pillars in 2D image
    boxes_2d = np.zeros((0,8,2))
    pillar_dim = self.opt.pillar_dims

    # Rotation ry is not really relevant for pillars
    v = np.dot(np.eye(3), np.array([1,0,0]))
    ry = -np.arctan2(v[2], v[0]) # i.e. ry = 0

    for i, center in enumerate(pc_3d.copy()[:3,:].T): # copy needed here to avoid messing up the original pc_3d
      # Flip center[0] if image flipped
      # This is required since img_info['calib'] refers to unflipped image and
      # introduces error when used on flipped image
      center[0] = center[0]*-1 if flipped else center[0]

      # Create a 3D pillar at pc location for the full-size image
      box_3d = compute_box_3d(dim=pillar_dim, location=center, rotation_y=ry)
      box_2d = project_to_image(box_3d, img_info['calib']).T  # [2x8]        
      
      # Flip box back if image flipped
      box_2d[1,:] = img_info['width']-box_2d[1,:]-1 if flipped else box_2d[1,:]

      # save the box for debug plots
      if self.opt.debug:
        box_2d_img, m = self._transform_pc(box_2d, inp_trans, self.opt.input_w, 
                                            self.opt.input_h, filter_out=False)
        boxes_2d = np.concatenate((boxes_2d, np.expand_dims(box_2d_img.T,0)),0)

      # Transform points to output image size, including data augmentation
      box_2d_t, m = self._transform_pc(box_2d, out_trans, self.opt.output_w, self.opt.output_h)
      
      # If transformed pillar box is only up to 1 pixel wide ignore it 
      if box_2d_t.shape[1] <= 1:
        continue

      # Get the bounding box in [xyxy] format
      bbox = [np.min(box_2d_t[0,:]), 
              np.min(box_2d_t[1,:]), 
              np.max(box_2d_t[0,:]), 
              np.max(box_2d_t[1,:])] # format: xyxy

      # Store height and width of the 2D box
      pillar_wh[0,i] = bbox[2] - bbox[0]
      pillar_wh[1,i] = bbox[3] - bbox[1]

    ## DEBUG ############################################################
    if self.opt.debug:
      img_2d = copy.deepcopy(img)
      # img_3d = copy.deepcopy(img)
      img_2d_inp = cv2.warpAffine(img, inp_trans, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
      img_2d_out = cv2.warpAffine(img, out_trans, 
                        (self.opt.output_w, self.opt.output_h),
                        flags=cv2.INTER_LINEAR)
      img_3d = cv2.warpAffine(img, inp_trans, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
      blank_image = 255*np.ones((self.opt.input_h,self.opt.input_w,3), np.uint8)
      overlay = img_2d_inp.copy()
      output = img_2d_inp.copy()

      pc_inp, _= self._transform_pc(pc_2d, inp_trans, self.opt.input_w, self.opt.input_h)
      pc_out, _= self._transform_pc(pc_2d, out_trans, self.opt.output_w, self.opt.output_h)

      pill_wh_inp = pillar_wh * (self.opt.input_w/self.opt.output_w)
      pill_wh_out = pillar_wh
      pill_wh_ori = pill_wh_inp * 2
      
      for i, p in enumerate(pc_inp[:3,:].T):
        color = int((p[2].tolist()/60.0)*255)
        color = (0,color,0)
        
        rect_tl = (np.min(int(p[0]-pill_wh_inp[0,i]/2), 0), np.min(int(p[1]-pill_wh_inp[1,i]),0))
        rect_br = (np.min(int(p[0]+pill_wh_inp[0,i]/2), 0), int(p[1]))
        cv2.rectangle(img_2d_inp, rect_tl, rect_br, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        img_2d_inp = cv2.circle(img_2d_inp, (int(p[0]), int(p[1])), 3, color, -1)

        ## On original-sized image
        rect_tl_ori = (np.min(int(pc_2d[0,i]-pill_wh_ori[0,i]/2), 0), np.min(int(pc_2d[1,i]-pill_wh_ori[1,i]),0))
        rect_br_ori = (np.min(int(pc_2d[0,i]+pill_wh_ori[0,i]/2), 0), int(pc_2d[1,i]))
        cv2.rectangle(img_2d, rect_tl_ori, rect_br_ori, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        img_2d = cv2.circle(img_2d, (int(pc_2d[0,i]), int(pc_2d[1,i])), 6, color, -1)
        
        p2 = pc_out[:3,i].T
        rect_tl2 = (np.min(int(p2[0]-pill_wh_out[0,i]/2), 0), np.min(int(p2[1]-pill_wh_out[1,i]),0))
        rect_br2 = (np.min(int(p2[0]+pill_wh_out[0,i]/2), 0), int(p2[1]))
        cv2.rectangle(img_2d_out, rect_tl2, rect_br2, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        img_2d_out = cv2.circle(img_2d_out, (int(p[0]), int(p[1])), 3, (255,0,0), -1)
        
        # on blank image
        cv2.rectangle(blank_image, rect_tl, rect_br, color, -1, lineType=cv2.LINE_AA)
        
        # overlay
        alpha = 0.1
        cv2.rectangle(overlay, rect_tl, rect_br, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # plot 3d pillars
        img_3d = draw_box_3d(img_3d, boxes_2d[i].astype(np.int32), [114, 159, 207], 
                    same_color=False)

      cv2.imwrite((self.opt.debug_dir+ '/{}pc_pillar_2d_inp.' + self.opt.img_format)\
        .format(self.img_ind), img_2d_inp)
      cv2.imwrite((self.opt.debug_dir+ '/{}pc_pillar_2d_ori.' + self.opt.img_format)\
        .format(self.img_ind), img_2d)
      cv2.imwrite((self.opt.debug_dir+ '/{}pc_pillar_2d_out.' + self.opt.img_format)\
        .format(self.img_ind), img_2d_out)
      cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_2d_blank.'+ self.opt.img_format)\
        .format(self.img_ind), blank_image)
      cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_2d_overlay.'+ self.opt.img_format)\
        .format(self.img_ind), output)
      cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_3d.'+ self.opt.img_format)\
        .format(self.img_ind), img_3d)
      self.img_ind += 1
    ## DEBUG ############################################################

    return pillar_wh


  def _flip_pc(self, pc_2d, width):
    """
    Flip pointcloud that is projected into the image.
    Invert the x-axis (in camera CS) by starting
    to count at the opposite edge of the image plane.
    
    :param pc_2d: np array [3xN]. Pointcloud projected into the
                                  image with depth as channel
                                  information.
    :param width: int. Width of input image.

    :return: pc_2d np array [3xN]. Flipped pointcloud in image.
    """
    pc_2d[0,:] = width - 1 - pc_2d[0,:] # Pixel counting starts at 1
    return pc_2d
  


  def _transform_pc(self, pc_2d, trans, img_width, img_height, filter_out=True):
    """
    Transform points from being projected into an image with input 
    image size to an image with output image size. The depth
    channel is not transformed.
    
    :param pc_2d: np array [3xN]. Pointcloud projected to image with input 
                                  image size. Depth as channel of "image".
    :param trans: np array [2x3]. Transformation from input shape to 
                                  output shape.
    """
    if pc_2d.shape[1] == 0: # If there is no y-axis return input
      return pc_2d, []

    pc_t = np.expand_dims(pc_2d[:2,:].T, 0) # [3,N]->[2,N]->[N,2]->[1,N,2]
    # Transform pointcloud projected into image with size of 
    # input image to an image with size of the output image.
    # Leave depth channel information untouched.
    t_points = cv2.transform(pc_t, trans)  
    t_points = np.squeeze(t_points,0).T     # [1,N,2]->[N,2]->[2,N]
    
    # Remove points outside image
    if filter_out:
      mask =    (t_points[0,:] < img_width) \
              & (t_points[1,:] < img_height) \
              & (0 < t_points[0,:]) \
              & (0 < t_points[1,:])
      # Mask and add depth channel to transformed "image" of projected pc
      out = np.concatenate((t_points[:,mask], pc_2d[2:,mask]), axis=0)
    else:
      mask = None
      out = np.concatenate((t_points, pc_2d[2:,:]), axis=0)

    return out, mask

  def _get_input_ef(self, img, trans_input):
    """
    Augment the radar projection for EF

    """
    # Resize / crop image
    inp = cv2.warpAffine(img.transpose((1,2,0)), 
                         trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    # cv2 expects a different format then pc_ef has
    inp = inp.transpose((2,0,1))

    return inp
  
  def _get_output_proj(self, img, trans_output):
    """
    Augment the radar projection for EF

    """
    # Resize / crop image
    inp = cv2.warpAffine(img.transpose((1,2,0)), 
                         trans_output, 
                         (self.opt.output_w, self.opt.output_h),
                         flags=cv2.INTER_LINEAR)

    return inp

  ## Augment, resize and normalize the image
  def _get_input(self, img, trans_input):
    """
    """
    # Resize / crop image
    inp = cv2.warpAffine(img, trans_input, 
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
    
    inp = (inp.astype(np.float32) / 255.)

    # Color augmentation
    if 'train' in self.split and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    # Normalize image
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    return inp


  def _init_ret(self, ret, gt_det):
    max_objs = self.max_objs * self.opt.dense_reg
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), 
      np.float32)
    ret['ind'] = np.zeros((max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((max_objs), dtype=np.float32)
    
    # Save rot_y and location for efficient snap generation
    ret['location'] = np.zeros((max_objs, 3), dtype=np.float32)
    ret['rotation_y'] = np.zeros((max_objs, 1), dtype=np.float32)
        
    if self.opt.use_lfa and ('train' in self.split or (not self.opt.no_lfa_val_loss)):
      ret['snaps'] = np.zeros((max_objs, self.opt.lfa_channel_in, self.opt.snap_resolution, self.opt.snap_resolution), dtype=np.float32)
      ret['frustum_bounds'] = np.zeros((max_objs, 2), dtype=np.float32)
      ret['nr_frustum_points'] = np.zeros((max_objs, 1), dtype=np.float32)
      if (self.opt.limit_use_closest or self.opt.limit_use_vel) and \
          self.opt.limit_frustum_points > 0:
        ret['alternative_point'] = np.zeros((max_objs, 8), dtype=np.float32)

    if self.opt.pointcloud:
      # ret['pc_box_hm'] are the output heatmaps with the required features 
      # filled in the 2D bbox on the image
      ret['pc_box_hm'] = np.zeros(
        (len(self.opt.pc_feat_lvl), self.opt.output_h, self.opt.output_w), 
        np.float32)
      
    regression_head_dims = {
      'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodel': 4, 
      'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2, 
      'dep': 1, 'dim': 3, 'amodel_offset': 2 }

    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        if head == 'nuscenes_att':
          ret[head + '_mask'] = np.zeros(
            (max_objs, regression_head_dims[head]), dtype=np.float32)
        else:
          ret[head + '_mask'] = np.zeros(
            (max_objs, 1), dtype=np.float32) # most masks only need 1 value
        gt_det[head] = []

    # LFA Net targets
    if self.opt.use_lfa:
      ret['pc_lfa_feat'] = np.zeros(
          (max_objs, len(self.opt.pc_feat_lvl)), dtype=np.float32)
      ret['lfa_mask'] = np.zeros(
            (max_objs, 1), dtype=np.float32) # most masks only need 1 value
      ret['bbox'] = np.zeros(
          (max_objs, 4), dtype=np.float32)
      ret['bbox_amodel'] = np.zeros(
          (max_objs, 4), dtype=np.float32)
        
    if 'hm_hp' in self.opt.heads:
      num_joints = self.num_joints
      ret['hm_hp'] = np.zeros(
        (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
      ret['hm_hp_mask'] = np.zeros(
        (max_objs * num_joints), dtype=np.float32)
      ret['hp_offset'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
      ret['hp_offset_mask'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)
    
    if 'rot' in self.opt.heads:
      ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
      ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
      ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
      gt_det.update({'rot': []})


  def _get_calib(self, img_info, width, height):
    """
    Get camera matrix (3x4).
    If it is given in image metadata us that. 
    If not use default calbration with focal length and image dimensions.
    """
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib

  def _scale_calib(self, calib, image_w, image_h, input_w, input_h):
    """
    Scale the camera matrix [3x4] to work on the input image size.
    The original camera matrix is scaled such that it works on the original,
    not scaled images. 
    Since P = K*[R t] this step is only correct if R = eye(3) and t = [0 0 0].T
    This is the case for CenterFusion, bus is checked anyways

    :param calib: Original calibration matrix P
    :param image_w: int Width of the unscaled image in pixels
    :param image_h: int height of the unscaled image in pixels 
    :param input_w: int Width of the image used as input into the backbone in pixels
    :param input_h: int height of the image used as input into the backbone in pixels

    """
    # Check R = eye(3) and t = [0 0 0].T (see function header)
    # Check last column of P and lower left triangular matrix for non zero elements
    if not (calib[:,3].nonzero()[0].shape[0] + calib[[0,1,2],[1,0,0]].nonzero()[0].shape[0]) == 0:
      raise ValueError("Calibration matrix can not be scaled using simple multiplication.")

    # Scale the focal length and principal points

    scale_x = input_w/image_w # Scaling for width / x dimension
    scale_y = input_h/image_h # Scaling for height / y dimension
    
    calib_input = copy.copy(calib)

    # Scale x components of image calibration matrix
    calib_input[0,[0,2]] *= scale_x
    # Scale y components of image calibration matrix
    calib_input[1,[1,2]] *= scale_y

    return calib_input
    

  def _ignore_region(self, region, ignore_val=1):
    np.maximum(region, ignore_val, out=region)


  def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
    # mask out crowd region, only rectangular mask is supported
    if cls_id == 0: # ignore all classes
      self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                        int(bbox[0]): int(bbox[2]) + 1])
    else:
      # mask out one specific class
      self._ignore_region(ret['hm'][abs(cls_id) - 1, 
                                    int(bbox[1]): int(bbox[3]) + 1, 
                                    int(bbox[0]): int(bbox[2]) + 1])
    if ('hm_hp' in ret) and cls_id <= 1:
      self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                          int(bbox[0]): int(bbox[2]) + 1])


  def _coco_box_to_bbox(self, box):
    """
    Compute bbox from coco annotation format of bounding boxes.
    There the width and height are given relative to the left vertices.
    Here the corners are computed in absolute values.
    Only 2 corner points with x,y needed since the bounding box is 
    axis aligned. 
    """
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _bbox_to_coco_box(self, bbox):
    """
    Transforms bounding box in [x1,y1,x2,y2] format into coco format [x,y,width,height]
    """
    coco_box = np.array([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
    return coco_box

  def _get_bbox_output(self, bbox, trans_output, height, width):
    """
    Compute axis aligned bounding box in frame and complete (amodel).
    "Amodel" because it possibly cannot be measured by the camera.
    Bounding boxes are transformed according to augmentation and output
    image resolution.
    Only 2 corner points with x,y needed per bounding box since the 
    bounding box is axis aligned. 
    """
    bbox = self._coco_box_to_bbox(bbox).copy() # from relative width height deifinition to absolute values

    # Compute actual rectangle from the 2 corners 
    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)

    # Scale rectangle down corresponding to output resolution + image augmentation
    for t in range(4):
      rect[t] =  affine_transform(rect[t], trans_output)
    bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
    bbox[2:] = rect[:, 0].max(), rect[:, 1].max()
    
    # complete bounding box (possibly partially outside of frame)
    bbox_amodel = copy.deepcopy(bbox)

    # clip bounding box to frame
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

    return bbox, bbox_amodel


  def _add_instance(
    self, ret, gt_det, k, cls_id, bbox, bbox_amodel, ann, trans_output,
    aug_s, calib, pre_cts=None, track_ids=None):
    """
    :param ret: Dictonary that will be "returned"
    :param gt_det: ground truth detection for debugger
    :param k: index of annotation
    :param cls_id: class id
    :param bbox: 2D bounding box
    :param bbox_amodel: amodel 2D bounding box 
    :param ann: annotation
    :param trans_output: Transformation matrix to output size
    :param aug_s: augumentation (?) scaling factor
    :param calib: Untransformed camera matrix
    :param pre_cts: Used for tracking?
    :param track_ids: ?
    
    """
    # Compute height and width of 2D bbox
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if h <= 0 or w <= 0: 
      return 
    
    # Compute object's true 2D center point in image    
    ct = np.array(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
    ct_int = ct.astype(np.int32)

    # Indices that the features/head outputs are filtered by.
    # Interpretation: At which pixel the center point is in the output sized image
    # The image is counted through row by row increasing in index count. Thus we can 
    # compute the index count by multiplying the row count (ct_int[1]) by the width of the
    # image (output_w). Then we need to go through the last row with the count on columns
    # (ct_int[0]).
    
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0] 
    
    # Create ground-truth heatmap with Gaussian kernels (that are limited by a radius)
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius)) 
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius) 
    gt_det['scores'].append(1) # Meta count for annotations
    
    ## Set up targets for each head for loss functions
    ## - define what is the target value for this category and how many parameters need to be defined
    ## - set up masks to only include annotations of sample (?) that have a target value for this head labeled. If there is no target value a default one is computed but due to mask then ignored
    
    # Classification
    ret['cat'][k] = cls_id - 1
    ret['mask'][k] = 1 # general mask

    # Center point for heatmap
    gt_det['cts'].append(ct)
    gt_det['clses'].append(cls_id - 1)

    # Regression
    ret['reg'][k] = ct - ct_int
    ret['reg_mask'][k] = 1

    # Width & Height of 2D bbox
    if 'wh' in ret:
      ret['wh'][k] = 1. * w, 1. * h
      ret['wh_mask'][k] = 1
    gt_det['bboxes'].append(
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))

    # Depth of centerpoint in 3D
    if 'dep' in self.opt.heads:
      if 'depth' in ann:
        ret['dep_mask'][k] = 1
        ret['dep'][k] = ann['depth'] * aug_s
        gt_det['dep'].append(ret['dep'][k])
      else:
        gt_det['dep'].append(2)
    
    # Local offset of centerpoint in img due to down-/upsampling
    if 'amodel_offset' in self.opt.heads:
      if 'amodel_center' in ann:
        # Transform to output size image
        amodel_center = affine_transform(ann['amodel_center'], trans_output)
        ret['amodel_offset_mask'][k] = 1
        # Compute offset in image
        ret['amodel_offset'][k] = amodel_center - ct_int
        gt_det['amodel_offset'].append(ret['amodel_offset'][k])
      else:
        gt_det['amodel_offset'].append([0, 0])

    # Dimensions of 3D bbox
    if 'dim' in self.opt.heads:
      if 'dim' in ann:
        ret['dim_mask'][k] = 1
        ret['dim'][k] = ann['dim']
        gt_det['dim'].append(ret['dim'][k])
      else:
        gt_det['dim'].append([1,1,1])

    # Rotation (yaw angle) of 3D bbox [alpha]
    if 'rot' in self.opt.heads:
      # ret contains the delta angle
      # gt_det contains the sin and cos of the delta angle and additioanl classification parameter
      self._add_rot(ret, ann, k, gt_det)

    # Save rot_y in gt for snapshot generation [rotation around y-axis of camera]
    ret['rotation_y'][k] = ann['rotation_y']
    # Save location in gt for efficient snapshot generation
    ret['location'][k] = ann['location']

    # Velocity (vx, vz) of 3D bbox
    if 'velocity' in self.opt.heads:
      if ('velocity_cam' in ann) and min(ann['velocity_cam']) > -1000:
        ret['velocity'][k] = np.array(ann['velocity_cam'], np.float32)[:3]
        ret['velocity_mask'][k] = 1
      gt_det['velocity'].append(ret['velocity'][k])

    # Nuscenes attribute of class (to further describe subtle differences in the class)
    if 'nuscenes_att' in self.opt.heads:
      if ('attributes' in ann) and ann['attributes'] > 0:
        att = int(ann['attributes'] - 1)
        ret['nuscenes_att'][k][att] = 1
        ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
      gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

    # Radar Heatmap in img with 2D bboxes filled with radar data of closest radar point to camera
    if self.opt.pointcloud:
      phase = 'train' if 'train' in self.split else 'val'
      # Calculate different distance thresholds for debug plots
      if self.opt.eval_frustum > 0:
        ct_amodel = ann['amodel_center']
        # Nabati distance threshold with error
        self.eval_frustum.dist_thresh_nabati = get_dist_thresh_nabati(calib, ct_amodel, ann['dim'], ann['alpha']) 
        opt_temp = copy.deepcopy(self.opt)
        opt_temp.use_dist_for_frustum = False
        self.eval_frustum.dist_thresh_depth = get_dist_thresh(calib, ct_amodel, ann['dim'], ann['alpha'], opt_temp, phase, ann['location'], ann['rotation_y']) 
        opt_temp.use_dist_for_frustum = True
        self.eval_frustum.dist_thresh_dist  = get_dist_thresh(calib, ct_amodel, ann['dim'], ann['alpha'], opt_temp, phase, ann['location'], ann['rotation_y']) 

      ## get pointcloud heatmap
      if self.opt.disable_frustum:
        # Disabled Frustum association -> simplified
        ret['pc_box_hm'] = ret['pc_hm']
        if self.opt.normalize_depth:
          ret['pc_box_hm'][self.opt.pc_feat_channels['pc_dep']] /= self.opt.max_pc_depth
      elif not self.opt.use_lfa:
        ## FRUSTUM CREATION WITH GROUND TRUTH FOR TRAINING ##
        # Get center point in original image size from amodel bbox.
        # Use amodel here since we compute the rotation for a 3D object
        
        dist_thresh = get_dist_thresh(None, None, ann['dim'], None, self.opt, phase=phase, location=ann['location'], rot_y=ann['rotation_y'])

        # Generate the feature heatmaps (distance, velocity in x and velocity in z)
        # Fill in 2D bbox with data of associated radar point

        pc_hm_to_box(ret['pc_box_hm'], ret['pc_hm'], ret['pc_hm_add'], ann, bbox, dist_thresh,  self.opt, self.eval_frustum)

      else: 
        # Use LFANet
        if (phase == 'train' or (not self.opt.no_lfa_val_loss)):
          # Compute dist thresh
          dist_thresh = get_dist_thresh(None, None, ann['dim'], None, self.opt, phase=phase, location=ann['location'], rot_y = ann['rotation_y'])

          # Differ between different snap generation methods
          pc_3d = copy.deepcopy(ret['pc_3d'][:,0:ret['pc_N']])
          
          # Generate snapshot of object
          if self.opt.snap_method == 'BEV':
            snap, frustum_bounds, nr_frustum_points, alternative_point = generate_snap_BEV(pc_3d, ann, dist_thresh, self.opt)
          elif self.opt.snap_method == 'proj':
            snap, frustum_bounds, nr_frustum_points, alternative_point = generate_snap_proj(pc_3d, ret['pc_snap_proj'], ann, bbox, dist_thresh, \
                                                                  ret['trans_original'], self.opt, calib)

          if self.opt.eval_frustum == 5:
            # Analyze created snapshot for number of points etc.
            self.eval_frustum.eval_snapshot(snap.copy(), frustum_bounds.copy(), copy.copy(nr_frustum_points), copy.copy(ann))


          ret['snaps'][k] = snap
          ret['frustum_bounds'][k] = np.array(frustum_bounds)
          ret['nr_frustum_points'][k] = nr_frustum_points
          if (self.opt.limit_use_closest or self.opt.limit_use_vel) and self.opt.limit_frustum_points > 0 and nr_frustum_points > 0:
            ret['alternative_point'][k] = alternative_point

        # Create Targets, fill GT heatmap if training LFANet and sec. heads separately
        if ret['velocity_mask'][k] != 0 and ret['dep_mask'][k] != 0:
          dep = ret['dep'][k]
          # Map gt vel into gt radial vel since we want to learn the radar point with the 
          # 'best' radial velocity. The transformation to orientated velocity is done in 
          # the secondary regression head
          gt_rad_vel = vgt_to_vrad(ret['velocity'][k][[0,2]], np.array(ann['location'])[[0,2]]) 
          rcs = None # no target for RCS
          # Target for LFANet
          ret['lfa_mask'][k] = 1 
          ret['pc_lfa_feat'][k] = [dep, gt_rad_vel[0], gt_rad_vel[1]] # don't norm dep by max dep for target!
          
          ret['bbox'][k] = bbox
          ret['bbox_amodel'][k] = bbox_amodel
          # Ground truth for pc BB hm in Learning Frustum Association as input for sec heads in training
          # It differs to the other pc BB hm computed for CF in using the gt depth and velocity from the center 
          # point directly and not from the pc
          # Annotations are written into pc_box_hm consecutively and they can overwrite each other.
          if ((self.opt.train_with_gt or (self.opt.lfa_with_ann and not self.opt.lfa_forward_to_sec)) and phase=='train') or (self.opt.eval_with_gt and phase=='val'): # Fill in pc box hm with annotation values and not pred from LFA. To train sec heads & LFANet separately
            if (self.opt.limit_use_closest or self.opt.limit_use_vel) and self.opt.limit_frustum_points > 0:
              set_pc_box_hm(ret['pc_box_hm'], bbox, bbox_amodel, dep, gt_rad_vel, rcs, nr_frustum_points, ret['alternative_point'][k], self.opt)
            else:
              set_pc_box_hm(ret['pc_box_hm'], bbox, bbox_amodel, dep, gt_rad_vel, rcs, nr_frustum_points, None, self.opt)

    # Tracking task
    if 'tracking' in self.opt.heads:
      if ann['track_id'] in track_ids:
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        ret['tracking_mask'][k] = 1
        ret['tracking'][k] = pre_ct - ct_int
        gt_det['tracking'].append(ret['tracking'][k])
      else:
        gt_det['tracking'].append(np.zeros(2, np.float32))

    # For MOT dataset
    if 'ltrb' in self.opt.heads:
      ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
        bbox[2] - ct_int[0], bbox[3] - ct_int[1]
      ret['ltrb_mask'][k] = 1

    # ltrb_amodel is to use the left, top, right, bottom bounding box representation 
    # to enable detecting out-of-image bounding box (important for MOT datasets)
    if 'ltrb_amodel' in self.opt.heads:
      ret['ltrb_amodel'][k] = \
        bbox_amodel[0] - ct_int[0], bbox_amodel[1] - ct_int[1], \
        bbox_amodel[2] - ct_int[0], bbox_amodel[3] - ct_int[1] 
      ret['ltrb_amodel_mask'][k] = 1
      gt_det['ltrb_amodel'].append(bbox_amodel)

    if 'hps' in self.opt.heads:
      self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

  def _add_hps(self, ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w):
    num_joints = self.num_joints
    pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3) \
        if 'keypoints' in ann else np.zeros((self.num_joints, 3), np.float32)
    if self.opt.simple_radius > 0:
      hp_radius = int(self.opt.simple_radius(h, w, min_overlap=self.opt.simple_radius))
    else:
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))

    for j in range(num_joints):
      pts[j, :2] = affine_transform(pts[j, :2], trans_output)
      if pts[j, 2] > 0:
        if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_w and \
          pts[j, 1] >= 0 and pts[j, 1] < self.opt.output_h:
          ret['hps'][k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
          ret['hps_mask'][k, j * 2: j * 2 + 2] = 1
          pt_int = pts[j, :2].astype(np.int32)
          ret['hp_offset'][k * num_joints + j] = pts[j, :2] - pt_int
          ret['hp_ind'][k * num_joints + j] = \
            pt_int[1] * self.opt.output_w + pt_int[0]
          ret['hp_offset_mask'][k * num_joints + j] = 1
          ret['hm_hp_mask'][k * num_joints + j] = 1
          ret['joint'][k * num_joints + j] = j
          draw_umich_gaussian(
            ret['hm_hp'][j], pt_int, hp_radius)
          if pts[j, 2] == 1:
            ret['hm_hp'][j, pt_int[1], pt_int[0]] = self.ignore_val
            ret['hp_offset_mask'][k * num_joints + j] = 0
            ret['hm_hp_mask'][k * num_joints + j] = 0
        else:
          pts[j, :2] *= 0
      else:
        pts[j, :2] *= 0
        self._ignore_region(
          ret['hm_hp'][j, int(bbox[1]): int(bbox[3]) + 1, 
                          int(bbox[0]): int(bbox[2]) + 1])
    gt_det['hps'].append(pts[:, :2].reshape(num_joints * 2))

  def _add_rot(self, ret, ann, k, gt_det):
    if 'alpha' in ann:
      ret['rot_mask'][k] = 1
      alpha = ann['alpha']
      # Check if ground truth angle is in first bin and if so set target bin and 
      # target delta angle to center angle.
      # First bin: [-7/6 pi, 1/6 pi]
      # First bin: [-1/6 pi, 7/6 pi]
      # Setting bin target to 1 means, if the angle is in bin i the second bin
      # classification variable is "active". "Active" means in this case that
      # the angle is in THIS bin. I.e. the bins refer to themselves in the 
      # second classification variable. rot has 6 variables and thus rot[:,1] is
      # giving the probability to be in bin 1. rot[:,0] the probability to be in 
      # bin 2. rot[:,4] the probability to be in bin 1. rot[:,5] the probability
      # to be in bin 2. 
      if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
        ret['rotbin'][k, 0] = 1
        ret['rotres'][k, 0] = alpha - (-0.5 * np.pi)  # center angle for first bin is -1/2 pi
      if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
        ret['rotbin'][k, 1] = 1
        ret['rotres'][k, 1] = alpha - (0.5 * np.pi) 
      # Catch if alpha is not in either bin (should not occur since bins cover the whole angle space)
      if not(alpha < np.pi / 6. or alpha > 5 * np.pi / 6.) and \
         not (alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.):
         raise ValueError("Ground truth angle is not in any bin! Alpha: ", alpha)
      # Store ground truth (c_i, sin(delta_theta_i), cos(delta_theta_i)) [see CenterNet Appendix] 
      gt_det['rot'].append(self._alpha_to_8(alpha))
    else:
      gt_det['rot'].append(self._alpha_to_8(0))
    
  def _alpha_to_8(self, alpha):
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    # Check again if angle is in bins
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1  # ret[1] stays 0 | value 1 at index 1 means the classifier in bin 1 predicts to 100% that the angle is in bin 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1  # ret[4] stays 0 | value 1 at index 5 means the classifier in bin 2 predicts to 100% that the angle is in bin 2
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
  
  def _format_gt_det(self, gt_det):
    if (len(gt_det['scores']) == 0):
      gt_det = {'bboxes': np.array([[0,0,1,1]], dtype=np.float32), 
                'scores': np.array([1], dtype=np.float32), 
                'clses': np.array([0], dtype=np.float32),
                'cts': np.array([[0, 0]], dtype=np.float32),
                'pre_cts': np.array([[0, 0]], dtype=np.float32),
                'tracking': np.array([[0, 0]], dtype=np.float32),
                'bboxes_amodel': np.array([[0, 0]], dtype=np.float32),
                'hps': np.zeros((1, 17, 2), dtype=np.float32),}
    gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
    return gt_det

  def fake_video_data(self):
    self.coco.dataset['videos'] = []
    for i in range(len(self.coco.dataset['images'])):
      img_id = self.coco.dataset['images'][i]['id']
      self.coco.dataset['images'][i]['video_id'] = img_id
      self.coco.dataset['images'][i]['frame_id'] = 1
      self.coco.dataset['videos'].append({'id': img_id})
    
    if not ('annotations' in self.coco.dataset):
      return

    for i in range(len(self.coco.dataset['annotations'])):
      self.coco.dataset['annotations'][i]['track_id'] = i + 1
