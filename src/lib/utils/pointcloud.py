from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import device_encoding
from turtle import color, pos
from matplotlib.colors import rgb2hex
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import RadarPointCloud
from functools import reduce
from typing import Tuple, Dict
from model.utils import _nms, _sigmoid, _topk, _tranpose_and_gather_feat
import os.path as osp
import torch
from typing import Tuple, List, Dict
import timeit
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

import copy

from utils.eval_frustum import EvalFrustum

from utils.ddd_utils import compute_corners_3d, compute_corners_3d_torch, alpha2rot_y, alpha2rot_y_torch, \
                            compute_box_3d, unproject_2d_to_3d_torch, compute_box_3d_torch
from utils.image import transform_preds_with_trans_torch

def map_pointcloud_to_image(pc, cam_intrinsic, img_shape=(1600,900)):
    """
    Map pointcloud from camera coordinates to the image.
    
    :param pc (PointCloud): point cloud in vehicle or global coordinates
    :param cam_cs_record (dict): Camera calibrated sensor record
    :param img_shape: shape of the image (width, height)
    :param coordinates (str): Point cloud coordinates ('vehicle', 'global') 
    :return: points (nparray), depth, mask: Mapped and filtered points with depth and mask
    """

    if isinstance(pc, RadarPointCloud):
        # Use only x,y,z of pc in camera coordinates and cut off the rest
        points = pc.points[:3,:] 
    else:
        points = pc

    (width, height) = img_shape
    depths = points[2, :] # z-coordinate in camera CS
    
    ## Take the actual picture
    points = view_points(points[:3, :], cam_intrinsic, normalize=True)

    ## Remove points that are either outside or behind the camera.
    # Note: Pixel counting starts at 1. 
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1) 
    mask = np.logical_and(mask, points[0, :] < width - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < height - 1)
    points = points[:, mask]
    # Overwrite depth information
    # Since x and y are scaled for z being scaled to 1
    # this overwriting means that the 3 dimensional point
    # in 'points' is no longer an actual point projected 
    # into a plane but instead it is an image point with 
    # additinal depth information.
    points[2,:] = depths[mask] 

    return points, mask


## A RadarPointCloud class where Radar velocity values are correctly 
# transformed to the target coordinate system
class RadarPointCloudWithVelocity(RadarPointCloud):
    # 0 | 1 | 2 | 3        | 4  | 5   | 6  | 7  | 8       | 9       
    # x | y | z | dyn_prop | id | rcs | vx | vy | vx_comp | vy_comp 
    # 10               | 11          | 12    | 13    | 14            | 15   | 16     | 17     | 18
    # is_quality_valid | ambig_state | x_rms | y_rms | invalid_state | pdh0 | vx_rms | vy_rms | dts
    
    @classmethod
    def rotate_velocity(cls, pointcloud, transform_matrix):
        n_points = pointcloud.shape[1]
        third_dim = np.zeros(n_points)
        pc_velocity = np.vstack((pointcloud[[8,9], :], third_dim, np.ones(n_points)))
        pc_velocity = transform_matrix.dot(pc_velocity)
        
        ## in camera coordinates, x is right, z is front
        pointcloud[[8,9],:] = pc_velocity[[0,2],:]

        return pointcloud

    @classmethod 
    def project_velocity(cls, pointcloud):
        """
        Projects the radial velocity directing towards the radar sensors towards the camera sensors.
        """
        #  6  | 7  | 8       | 9       | 16     | 17     | 18
        #  vx | vy | vx_comp | vy_comp | vx_rms | vy_rms | dts



    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 1.0,
                             invalid_states: List[int] = [0], # default value from nuScenes
                             dynprop_states: List[int] = range(7), # default value from nuScenes
                             ambig_states: List[int] = [3] # default value from nuScenes
                             ) -> Tuple['PointCloud', np.ndarray]:

        """
        !!! THIS METHOD IS TAKEN FROM THE nuSenes DEV KIT AND ADAPTED !!! (data_classes.py)

        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate system, we need to map the coordinates to a single reference system.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car CS to reference CS.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
        ref_from_car_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car CS.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)
        car_from_global_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']), 
                                        invalid_states,
                                        dynprop_states,
                                        ambig_states)
            current_pc.remove_close(min_distance) # nuScenes method

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)
            global_from_car_rot = transform_matrix([0.0, 0.0, 0.0],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate CS to ego car CS.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)
            car_from_current_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(current_cs_rec['rotation']), inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            velocity_trans_matrix = reduce(np.dot, [ref_from_car_rot, car_from_global_rot, global_from_car_rot, car_from_current_rot])
            current_pc.transform(trans_matrix)

            # Do the required rotations to the Radar velocity values
            # This only rotates the velocity so that they can be expressed in the camera CS. They are NOT
            # projected into the Center of the camera.
            current_pc.points = cls.rotate_velocity(current_pc.points, velocity_trans_matrix)

            # Project the velocities into the correct direction
            # current_pc.points = cls.project_velocity(current_pc.points)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        return all_pc, all_times


def get_alpha(rot):
  # Check which classification score is higher and
  # take classification with larger certainty
  idx = rot[:, 1] > rot[:, 5] 
  alpha1 = torch.atan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi) # angle from nin 1
  alpha2 = torch.atan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi) # angle from bin 2
  alpha = alpha1 * idx.float() + alpha2 * (~idx).float()
  return alpha



def get_dist_thresh(calib, ct, dim, alpha, opt, phase=None, location = None, rot_y=None):
    """
    Threshold for the radar / frustum association. 
    Size of the bounding box in the viewing direction of radar
    
    calib: Camera calibration matrix
    ct: Centerpoint of object in image (original coordinates!)
    dim: Dimensions of 3D bbox
    location: 3D position of 3D bbox
    alpha: absolute rotation angle 
    """
    if rot_y == None:
      rot_y = alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]) # - vor calib hier eigentlich nicht
    
    # Get expansion ratio depending on phase
    if phase == 'train':
      expansion_ratio = opt.frustumExpansionRatioTrain
    elif phase == 'val':
      expansion_ratio = opt.frustumExpansionRatioVal
    else:
      expansion_ratio = 0
    
    if not opt.use_dist_for_frustum: # USe depth for frustum
      # Conventional approach, calculate distance threshold as scalar with the average depth deviation
      corners_3d = compute_corners_3d(dim, rot_y)
      # Compute 2D bounding box corners in BEV
      # corners_2d = corners_3d[0:4][:,[0,2]]
      
      # Approach using the depth to the bounding box corners
      # Get dimensions of 3D bbox along z axis and compute the range
      dist_thresh = (max(corners_3d[:,2]) - min(corners_3d[:,2])) / 2.0 # dist_thresh is the range for radar points to lie inside the frustum
    
      # Apply Expansion ratio
      dist_thresh += dist_thresh * expansion_ratio

    else: # Use dist for frustum
      # Calculate distance threshold as vector of 2 values: upper and lower distance threshold
      # using the distance 
      corners_3d = compute_box_3d(dim, location, rot_y)
      # Compute 2D bounding box corners in BEV
      corners_2d = corners_3d[0:4][:,[0,2]]
      # Approach using the distance to the bounding box corners
      corners_norm = np.linalg.norm(corners_2d, axis = 1)
      dist_thresh = np.array([np.amin(corners_norm), np.amax(corners_norm)])
      # dist_thresh = (np.amax(corners_norm) - np.amin(corners_norm)) / 2.0

      # Subtract the ratio from the lower bound and add it to the upper bound
      dist_thresh += np.array([-1, 1]) * np.diff(dist_thresh)/2 * expansion_ratio
      dist_thresh[0] = max(0,dist_thresh[0]) # Make sure threshold is not negative

    return dist_thresh



def get_dist_thresh_torch(calib, ct, dim, alpha, opt, phase=None, location=None, rot_y=None, device=None):
  """
  Threshold for the radar / frustum association. 
  Size of the bounding box in the viewing direction of radar
  
  calib: Camera calibration matrix
  ct: Centerpoint of object in image
  dim: Dimensions of 3D bbox
  location: 3D position of 3D bbox
  alpha: absolute rotation angle 
  """
  if rot_y == None:
    rot_y = alpha2rot_y_torch(alpha, ct[0], calib[0, 2], calib[0, 0])
  
  # Get expansion ratio depending on phase
  if phase == 'train':
    expansion_ratio = torch.tensor(opt.frustumExpansionRatioTrain, device=device)
  elif phase == 'val':
    expansion_ratio = torch.tensor(opt.frustumExpansionRatioVal, device=device)
  else:
    expansion_ratio = torch.tensor(0, device=device)

  if not opt.use_dist_for_frustum: # Use depth for frustum

    # Conventional approach, calculate distance threshold as scalar with the average depth deviation    
    corners_3d = compute_corners_3d_torch(dim, rot_y, device=device)
    # Compute 2D bounding box corners in BEV
    # corners_2d = corners_3d[0:4][:,[0,2]]

    # Approach using the depth to the bounding box corners
    # Get dimensions of 3D bbox along z axis and compute the range
    dist_thresh = (torch.amax(corners_3d[:,2]) - torch.amin(corners_3d[:,2])) / 2.0 # dist_thresh is the range for radar points to lie inside the frustum
    
    # Apply Expansion ratio
    dist_thresh += dist_thresh * expansion_ratio

  else: # Use dist for frustum

    corners_3d = compute_box_3d_torch(dim, location, rot_y, device=device)
    # Compute 2D bounding box corners in BEV
    corners_2d = corners_3d[0:4][:,[0,2]]
    # Approach using the distance to the bounding box corners

    corners_norm = torch.linalg.norm(corners_2d, axis = 1)
    dist_thresh = torch.tensor([torch.amin(corners_norm), torch.amax(corners_norm)], device=device)
    
    #### dist_thresh = (torch.amax(corners_norm) - torch.amin(corners_norm)) / 2.0
    
    # Subtract the ratio from the lower bound and add it to the upper bound
    dist_thresh += torch.tensor([-1, 1], device=device) * torch.diff(dist_thresh)/2 * expansion_ratio
    dist_thresh[0] = max(0,dist_thresh[0]) # Make sure threshold is not negative

  return dist_thresh



def get_dist_thresh_nabati(calib, ct, dim, alpha):
    """
    Threshold for the radar / frustum association. 
    Size of the bounding box in the viewing direction of radar
    
    !!! DO NOT USE THIS FOR TRAINING, THERE IS A BUG (Missing '()') !!!
    
    calib: Camera calibration matrix
    ct: Centerpoint of object
    dim: Dimensions of 3D bbox
    alpha: absolute rotation angle 
    """
    rotation_y = alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0])
    corners_3d = compute_corners_3d(dim, rotation_y)
    # Get dimensions of 3D bbox along z axis and compute the range
    dist_thresh = max(corners_3d[:,2]) - min(corners_3d[:,2]) / 2.0 # !!! MISSING '()'
    return dist_thresh


def get_rules_based(dep,pc_depth,pc_vx,pc_vz,pc_rcs):
  """
  Generate the pc box heatmap using a rules based approach.
  If only one point is in the ROI, no further measures are taken.
  If 2 points are in the ROI, the one close to the output's centerpoint is used.
  If > 2 points are in the ROI, they are sorted using the mean and std dev, then a mean is the output.

  :param dep: Object depth / estimated depth
  :param depth: list of filtered depth values 
  """
  if len(pc_depth) == 1:
    # Return index 0
    depth = pc_depth[0]
    vx = pc_vx[0]
    vz = pc_vz[0]
    rcs = pc_rcs[0]  
  elif len(pc_depth) == 2:
    # 2 points given. Choose the one closer to the distance estimation
    idx = np.argmin(np.abs(pc_depth-dep))
    depth = pc_depth[idx]
    vx = pc_vx[idx]
    vz = pc_vz[idx]
    rcs = pc_rcs[idx]
  else:
    # Calculate mean and std of summed velocity for every point
    abs_v = pc_vx+pc_vz
    std = np.std(abs_v)
    mean = np.mean(abs_v)
    # Calculate all indices within mean + std
    indices = (abs_v <= (mean+std)) & (abs_v >= (mean-std))
    
    if not np.any(indices):
      # Fallback: choose closest
      print("!!! NO INDEX SELECTED IN RULES BASED. CHOOSE CLOSEST !!!")
      indices = 0

    depth = np.mean(pc_depth[indices])
    vx = np.mean(pc_vx[indices])
    vz = np.mean(pc_vz[indices])
    rcs = np.mean(pc_rcs[indices])

  return depth, vx, vz, rcs


def get_rules_based_torch(dep,pc_depth,pc_vx,pc_vz,pc_rcs):
  """
  Generate the pc box heatmap using a rules based approach.
  If only one point is in the ROI, no further measures are taken.
  If 2 points are in the ROI, the one close to the output's centerpoint is used.
  If > 2 points are in the ROI, they are sorted using the mean and std dev, then a mean is the output.

  :param dep: Object depth / estimated depth
  :param depth: list of filtered depth values 
  """
  if len(pc_depth) == 1:
    # Return index 0
    depth = pc_depth[0]
    vx = pc_vx[0]
    vz = pc_vz[0]
    rcs = pc_rcs[0]  
  elif len(pc_depth) == 2:
    # 2 points given. Choose the one closer to the distance estimation
    idx = torch.argmin(torch.abs(pc_depth-dep))    
    depth = pc_depth[idx]
    vx = pc_vx[idx]
    vz = pc_vz[idx]
    rcs = pc_rcs[idx]
  else:
    # Calculate mean and std of summed velocity for every point
    abs_v = pc_vx+pc_vz
    std = torch.std(abs_v)
    mean = torch.mean(abs_v)
    # Calculate all indices within mean + std
    indices = (abs_v <= (mean+std)) & (abs_v >= (mean-std))

    if not torch.any(indices):
      # Fallback: choose closest
      print("!!! NO INDEX SELECTED IN RULES BASED. CHOOSE CLOSEST !!!")
      indices = 0
    
    depth = torch.mean(pc_depth[indices])
    vx = torch.mean(pc_vx[indices])
    vz = torch.mean(pc_vz[indices])
    rcs = torch.mean(pc_rcs[indices])

  return depth, vx, vz, rcs



def generate_pc_box_hm(output, pc_hm, pc_hm_add, calibs, opt, trans_originals):
    """
    Generates pointcloud heatmap when not in training.
    """
    phase = 'val'
    K = opt.K # max number of output objects
    heat = output['hm'].clone()
    heat = _sigmoid(heat) # convert scores to probabilities
    heat = _nms(heat)
    wh = output['wh']
    pc_box_hm = torch.zeros_like(pc_hm)  # init with zeros. if no radar point is inside frustum pc_box_hm is zero everywhere

    batch, cat, height, width = heat.size()
    scores, inds, clses, ys0, xs0 = _topk(heat, K=K) # get top K center points from heatmap (called peaks)
    scores = scores.view(batch, K, 1, 1)
    xs = xs0.view(batch, K, 1) + 0.5
    ys = ys0.view(batch, K, 1) + 0.5
    
    ## Initialize pc_feats
    pc_feats = torch.zeros((batch, len(opt.pc_feat_lvl), height, width), device=heat.device)
    dep_ind = opt.pc_feat_channels['pc_dep']
    vx_ind = opt.pc_feat_channels['pc_vx']
    vz_ind = opt.pc_feat_channels['pc_vz']
    to_log = opt.sigmoid_dep_sec
    
    ## Get top estimated depths
    out_dep = 1. / (output['dep'].sigmoid() + 1e-6) - 1.  # transform depths with Eigen et. al (essentially e^(-output['dep']))
    dep = _tranpose_and_gather_feat(out_dep, inds) # B x K x (C)
    if dep.size(2) == cat:
      cats = clses.view(batch, K, 1, 1)
      dep = dep.view(batch, K, -1, 1) # B x K x C x 1
      dep = dep.gather(2, cats.long()).squeeze(2) # B x K x 1 | only get the top K depth values over all classes and not per class

    ## Get top bounding boxes
    wh = _tranpose_and_gather_feat(wh, inds) # B x K x 2
    wh = wh.view(batch, K, 2)
    wh[wh < 0] = 0
    if wh.size(2) == 2 * cat: # cat spec
      wh = wh.view(batch, K, -1, 2)
      cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
      wh = wh.gather(2, cats.long()).squeeze(2) # B x K x 2 | only get the top K wh values over all classes and not per class
    # assemble boxes
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)  # B x K x 4
    
    ## Get top dimensions
    dims = _tranpose_and_gather_feat(output['dim'], inds).view(batch, K, -1)

    ## Get top rotation
    rot = _tranpose_and_gather_feat(output['rot'], inds).view(batch, K, -1)

    ## Calculate values for the new pc_box_hm
    clses = clses.cpu().numpy()

    ## Iterate over top K values (or batches) of all parameters over all classes
    for i, [pc_hm_b, pc_hm_add_b, bboxes_b, depth_b, dim_b, rot_b, score_b] in enumerate(zip(pc_hm, pc_hm_add, bboxes, dep, dims, rot, scores)):
      alpha_b = get_alpha(rot_b).unsqueeze(1)
      calib_b = calibs[i] #if calibs.shape[0] > 1 else calibs[0]
      trans_original_b = trans_originals[i] #if trans_originals.shape[0] > 1 else trans_originals[0]
      
      if opt.sort_det_by_depth: # sort such that farthest is first. Such that closest overwrides farther away point
        idx = torch.argsort(depth_b[:,0])
        bboxes_b = bboxes_b[idx,:]
        depth_b = depth_b[idx,:]
        dim_b = dim_b[idx,:]
        rot_b = rot_b[idx,:]
        alpha_b = alpha_b[idx,:]
        score_b = score_b[idx,:]

      ## Iterate over top K values (or batches) of all parameters over all classes
      for j, [bbox, depth, dim, alpha, score] in enumerate(zip(bboxes_b, depth_b, dim_b, alpha_b, score_b)):
        # If prediction is not accurate enough skip it to save runtime
        if score >= opt.lfa_pred_thresh:
          clss = clses[i,j].tolist()
          # Calculate ct in output image coordinates
          ct = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], device=pc_hm_b.device)
          # Calculate ct in original image size coordinates (to be able to apply camera matrix)
          ct_trans = transform_preds_with_trans_torch(ct, trans_original_b.clone())

          location = unproject_2d_to_3d_torch(ct_trans, depth, calib_b) if opt.use_dist_for_frustum else None
          dist_thresh = get_dist_thresh_torch(calib_b, ct_trans, dim, alpha, opt, phase, location)
          
          pc_hm_to_box_torch(pc_box_hm[i], pc_hm_b, pc_hm_add_b, depth, location, bbox, dist_thresh, opt) # project radar points into image and fill part of bbox with radar values in separate channels 

    return pc_box_hm


def pc_hm_to_box_torch(pc_box_hm, pc_hm, pc_hm_add, dep, location, bbox, dist_thresh, opt):
    """
    This function assigns the radar point to the detection with a frustum approach.
    Function analog to pc_hm_to_box but used in validation, implemented in torch.

    :param pc_box_hm: Heatmap with the selected features (from whole bbox) as channels. Only given as argument to be "returned"
    :param pc_hm: Radar Pillar heat map(s) with radar features to learn with
    :param pc_hm_add: Heat map containing the additional information (delta timestamps, position (x and y))
    :param ann: The annotation of the current object
    :param bbox: 2D bounding box dimensions
    :param dist_thresh: Distance threshold for frustum association calculated from 3D bounding box corner.
                        The dimensions of this param change whether opt.use_dist_for_frustum is True or not.
                        If False: dist_thresh is scalar, if True: dist_thresh is [2,] tensor describing lower and upper bound
    :param opt: Options
    :param eval_frustum: Object of class eval_frustum to save Frustum analyzation data
    """
    
    # Create dict with the feature levels as keys to store the features
    feature_values = dict.fromkeys(opt.pc_feat_lvl)
    use_rcs = opt.rcs_feature_hm # True if rcs should be added as additional feature heatmap

    if isinstance(dep, list) and len(dep) > 0:
      dep = dep[0]
    ct = torch.tensor(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=torch.float32)
    bbox_int = torch.tensor([torch.floor(bbox[0]), 
                         torch.floor(bbox[1]), 
                         torch.ceil(bbox[2]), 
                         torch.ceil(bbox[3])], dtype=torch.int32)# format: xyxy

    roi = pc_hm[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
    roi_additional = pc_hm_add[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
    roi = torch.cat((roi,roi_additional)) # Extend the roi with additional heatmap features
    pc_dep = roi[opt.pc_feat_channels['pc_dep']]
    pc_vx = roi[opt.pc_feat_channels['pc_vx']]
    pc_vz = roi[opt.pc_feat_channels['pc_vz']]
    pc_rcs = roi[opt.pc_feat_channels['pc_rcs']] if use_rcs else torch.zeros_like(pc_dep)
    # pc_dts = roi[-3] # Not used currently
    pc_pos_x = roi[-2] # Camera CS
    pc_pos_z = roi[-1] # Camera CS

    pc_dist = torch.sqrt(torch.square(pc_pos_x)+torch.square(pc_pos_z))
    nonzero_inds = torch.nonzero(pc_dep, as_tuple=True)

    if len(nonzero_inds) and len(nonzero_inds[0]) > 0:
    #   nonzero_pc_dep = torch.exp(-pc_dep[nonzero_inds])
      nonzero_pc_dep = pc_dep[nonzero_inds]
      nonzero_pc_vx = pc_vx[nonzero_inds]
      nonzero_pc_vz = pc_vz[nonzero_inds]
      nonzero_pc_rcs = pc_rcs[nonzero_inds]
      nonzero_pc_dist = pc_dist[nonzero_inds]

      if opt.use_dist_for_frustum:
        range_gt = np.linalg.norm(location[[0,2]])
        # within_thresh = (nonzero_pc_dist < range_gt + dist_thresh) \
        #               & (nonzero_pc_dist > max(0, range_gt - dist_thresh)) # boolean indexing
        within_thresh = (nonzero_pc_dist < dist_thresh[1]) \
              & (nonzero_pc_dist >  dist_thresh[0]) # boolean indexing
      else:
        # Standard approach by Nabati, using depth
        ## Get points within dist threshold, i.e. within frustum
        within_thresh = (nonzero_pc_dep < dep + dist_thresh) \
                & (nonzero_pc_dep > max(0, dep - dist_thresh)) # boolean indexing


      # DEBUG
      # within_thresh_depth = (nonzero_pc_dep < dep + dist_thres_depth) \
      #           & (nonzero_pc_dep > max(0, dep - dist_thres_depth)) # boolean indexing
      # if not torch.all(within_thresh==within_thresh_depth) and not torch.any(within_thresh):
      #   debug = 0

      pc_dep_match = nonzero_pc_dep[within_thresh]
      pc_vx_match = nonzero_pc_vx[within_thresh]
      pc_vz_match = nonzero_pc_vz[within_thresh]
      pc_rcs_match = nonzero_pc_rcs[within_thresh]

      if len(pc_dep_match) > 0:
        
        if not opt.use_rules_based:
          # Frustum association by selection the closest point to the radar sensor
          idx_selection = torch.argmin(pc_dep_match)
          depth = pc_dep_match[idx_selection]
          vx = pc_vx_match[idx_selection]
          vz = pc_vz_match[idx_selection]
          rcs = pc_rcs_match[idx_selection]
        else:
          # Get unique items
          pc_dep_match_u, inverse = torch.unique(pc_dep_match, return_inverse=True)
          perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
          inverse, perm = inverse.flip([0]), perm.flip([0])
          perm = inverse.new_empty(pc_dep_match_u.size(0)).scatter_(0, inverse, perm)
          pc_vx_match_u = pc_vx_match[perm]
          pc_vz_match_u = pc_vz_match[perm]
          pc_rcs_match_u = pc_rcs_match[perm]

          depth, vx, vz, rcs = get_rules_based_torch(dep, pc_dep_match_u, pc_vx_match_u, pc_vz_match_u, pc_rcs_match_u)

        if opt.normalize_depth:
          depth /= opt.max_pc_depth

        # Save feature values in feature_values
        feature_values['pc_dep'] = depth
        feature_values['pc_vx'] = vx
        feature_values['pc_vz'] = vz
        if use_rcs:
          feature_values['pc_rcs'] = rcs

        w = bbox[2] - bbox[0]
        w_interval = opt.hm_to_box_ratio*(w)
        w_min = int(ct[0] - w_interval/2.)
        w_max = int(ct[0] + w_interval/2.)
        
        h = bbox[3] - bbox[1]
        h_interval = opt.hm_to_box_ratio*(h)
        h_min = int(ct[1] - h_interval/2.)
        h_max = int(ct[1] + h_interval/2.)

        # Write the feature values into the heatmaps
        # fill all pixels of bbox projected into 3D space with the same data of associated 
        # radar point.
        for feat in opt.pc_feat_lvl:
          pc_box_hm[opt.pc_feat_channels[feat],
                h_min:h_max+1, 
                w_min:w_max+1+1] = feature_values[feat]

def pc_hm_to_box(pc_box_hm, pc_hm, pc_hm_add, ann, bbox, dist_thresh, opt, eval_frustum: EvalFrustum):
    """
    This function assigns the radar point "in a frustum" to the bounding box / heatmap!
    Essential!
    Used for training.
    
    :param pc_box_hm: Heatmap with the selected features (from whole bbox) as channels. Only given as argument to be "returned"
    :param pc_hm: Radar Pillar heat map(s) with radar features to learn with
    :param pc_hm_add: Heat map containing the additional information (delta timestamps, position (x and y))
    :param ann: The annotation of the current object
    :param bbox: 2D bounding box dimensions
    :param dist_thresh: Distance threshold for frustum association calculated from 3D bounding box corner.
                        The dimensions of this param change whether opt.use_dist_for_frustum is True or not.
                        If False: dist_thresh is scalar, if True: dist_thresh is [2,] np.array describing lower and upper bound
    :param opt: Options
    :param eval_frustum: Object of class eval_frustum to save Frustum analyzation data
    """

    # Create dict with the feature levels as keys to store the features
    feature_values = dict.fromkeys(opt.pc_feat_lvl)
    use_rcs = opt.rcs_feature_hm # True if rcs should be added as additional feature heatmap

    ct = np.array( # Calculate center point of bounding box
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) # center point
    bbox_int = np.array([np.floor(bbox[0]), 
                         np.floor(bbox[1]), 
                         np.ceil(bbox[2]), 
                         np.ceil(bbox[3])], np.int32) # format: xyxy | get conservative discretized estimate of bbox

    roi = pc_hm[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1] # Region of Interest corresponding to 2D detection
    roi_additional = pc_hm_add[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
    roi = np.concatenate((roi,roi_additional)) # Extend the roi with additional heatmap features
    pc_dep = roi[opt.pc_feat_channels['pc_dep']] # z component of position
    pc_vx = roi[opt.pc_feat_channels['pc_vx']]
    pc_vz = roi[opt.pc_feat_channels['pc_vz']]
    pc_rcs = roi[opt.pc_feat_channels['pc_rcs']] if use_rcs else np.zeros(pc_dep.shape)
    pc_dts = roi[-2]
    pc_pos_x = roi[-1] # Camera CS
    # Calc distance for each point in heatmap
    pc_dist = np.sqrt(np.square(pc_pos_x)+np.square(pc_dep))

    nonzero_inds = np.nonzero(pc_dep) # at these indices there is a valid radar point measurement
    
    if len(nonzero_inds[0]) > 0: # if there are any radar points
    #   nonzero_pc_dep = np.exp(-pc_dep[nonzero_inds])
      nonzero_pc_dep = pc_dep[nonzero_inds]
      nonzero_pc_vx = pc_vx[nonzero_inds]
      nonzero_pc_vz = pc_vz[nonzero_inds]
      nonzero_pc_rcs = pc_rcs[nonzero_inds]
      nonzero_pc_dts = pc_dts[nonzero_inds]
      nonzero_pc_pos_x = pc_pos_x[nonzero_inds]
      nonzero_pc_dist = pc_dist[nonzero_inds]

      if opt.use_dist_for_frustum:
        range_gt = np.linalg.norm(ann['location'][0::2])
        # within_thresh = (nonzero_pc_dist < range_gt + dist_thresh) \
        #               & (nonzero_pc_dist > max(0, range_gt - dist_thresh)) # boolean indexing
        within_thresh = (nonzero_pc_dist < dist_thresh[1]) \
                        & (nonzero_pc_dist > dist_thresh[0]) # boolean indexing
      else:
      # if True:
        dep_gt = ann['depth']
        if isinstance(dep_gt, list) and len(dep_gt) > 0:
          dep_gt = dep_gt[0]
        # Standard approach by Nabati, using depth
        ## Get points within dist threshold, i.e. within frustum
        within_thresh = (nonzero_pc_dep < dep_gt+dist_thresh) \
                & (nonzero_pc_dep > max(0, dep_gt-dist_thresh)) # boolean indexing
      
        
      pc_dep_match = nonzero_pc_dep[within_thresh]
      pc_vx_match = nonzero_pc_vx[within_thresh]
      pc_vz_match = nonzero_pc_vz[within_thresh]
      pc_rcs_match = nonzero_pc_rcs[within_thresh]
      pc_dts_match = nonzero_pc_dts[within_thresh]
      pc_pos_x_match = nonzero_pc_pos_x[within_thresh]

      if len(pc_dep_match) > 0: # if there are any radar points inside of the frustum
        
        if not opt.use_rules_based:
          # Frustum association by selection the closest point to the radar sensor
          idx_selection = np.argmin(pc_dep_match)
          depth = pc_dep_match[idx_selection]
          vx = pc_vx_match[idx_selection]
          vz = pc_vz_match[idx_selection]
          rcs = pc_rcs_match[idx_selection]
        else:
          # Get unique items
          pc_dep_match_u, indices_unique = np.unique(pc_dep_match, return_index=True)
          pc_vx_match_u = pc_vx_match[indices_unique]
          pc_vz_match_u = pc_vz_match[indices_unique]
          pc_rcs_match_u = pc_rcs_match[indices_unique]

          depth, vx, vz, rcs = get_rules_based(dep_gt, pc_dep_match_u, pc_vx_match_u, pc_vz_match_u, pc_rcs_match_u)

        # dts = pc_dts_match[idx_selection] # Not necessary for the 'closest' association method

        if opt.normalize_depth:
          depth /= opt.max_pc_depth

        # Save feature values in feature_values
        feature_values['pc_dep'] = depth
        feature_values['pc_vx'] = vx
        feature_values['pc_vz'] = vz
        if use_rcs:
          feature_values['pc_rcs'] = rcs
        
        # Get 2D bbox
        w = bbox[2] - bbox[0]
        w_interval = opt.hm_to_box_ratio*(w)
        w_min = int(ct[0] - w_interval/2.)
        w_max = int(ct[0] + w_interval/2.)
        
        h = bbox[3] - bbox[1]
        h_interval = opt.hm_to_box_ratio*(h)
        h_min = int(ct[1] - h_interval/2.)
        h_max = int(ct[1] + h_interval/2.)

        # Write the feature values into the heatmaps
        # fill all pixels of bbox projected into 3D space with the same data of associated 
        # radar point.
        for feat in opt.pc_feat_lvl:
          pc_box_hm[opt.pc_feat_channels[feat],
                h_min:h_max+1, 
                w_min:w_max+1+1] = feature_values[feat]

        ######################################## DEBUG #########################################

        # Analyze the selection of the frustum association

        if opt.eval_frustum > 0:
          eval_frustum.analyze_frustum_association(pc_pos_x_match, pc_dep_match,
                                                   pc_vx_match, pc_vz_match,                                                                                                                       
                                                   pc_rcs_match,
                                                   pc_dts_match,
                                                   idx_selection,
                                                   ann,
                                                   dist_thresh,
                                                   )


        ########################################################################################

