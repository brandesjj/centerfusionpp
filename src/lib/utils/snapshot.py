from math import log
import numpy as np
from pyparsing import rest_of_line
import torch
import matplotlib.pyplot as plt
import copy

from utils.ddd_utils import calc_frustum_corners_torch, calc_frustum_corners, compute_box_3d_torch, compute_box_3d, \
                      get_2d_rotation_matrix, get_2d_rotation_matrix_torch, project_to_image, project_to_image_torch


def generate_snap_proj(pc_3d, pc_snap_proj, obj, bbox, dist_thresh, trans_original, opt, calib, check_only=False):
  """
  This function generates the input image for the LFANet using the projection method.
  Advantage is that it should be way faster

  
  :param pc_hm: Heatmap with the selected features (from one point) as channels. Only given as argument to be "returned"
  :param pc_3d: tensor 3D pointcloud of the radar including all 18 images (filtered for points above distance threshold (see opts.py))    
  :param obj: Dict containing information of the obj regarded. Contains fields:
                  "location": Center of bounding box in 3D space [x,y,z] in camera CS
                  "rotation": Global orientation of obj around Y-axis of camera
                  "dimension": Dimension in X,Y and Z of 3D-BB
                  [optional] "cat": Category/class of obj
  :param dist_thresh: Distance threshold for frustum association calculated from 3D bounding box corner.
                      The dimensions of this param change whether opt.use_dist_for_frustum is True or not.
                      If False: dist_thresh is scalar, if True: dist_thresh is [2,] tensor describing lower and upper bound
  :param opt: dict Options
  :param check_only: Flag to save computational effort when training with gt heatmap. If True, return 1 
                     when there is a radar point inside the frustum and skip computing the actual snap!
                     Default: False
  :param trans_original: Transforms output size [200x112] back to original image size [1600x900]
  
  :return frustum_points: Return the frustum_points for filling in the features of the closest point (or any point) into pc_bb_hm when there are not enough point in the frustum for the LFANet to work with. It is None if either the snap is totally empty or has enough points for LFANet.
  """

  # Transform bounding box to original image sized coordinates
  bbox = bbox.copy().reshape(2,2)
  bbox = bbox@trans_original[0:2,0:2]
  bbox[0,:] += trans_original[:,2].T
  bbox[1,:] += trans_original[:,2].T
  bbox = bbox.flatten()

  snap_channels = opt.snap_channels
  output_size = opt.snap_resolution

  if opt.use_dist_for_frustum:
    # # Rotation approach.
    bound_l = dist_thresh[0]
    bound_u = dist_thresh[1]
    obj_range = np.linalg.norm(np.array((obj['location'][0],obj['location'][2])))

  else:
    # Approach without rotation to maximize size of frustum in snap
    obj_range = obj['location'][2]
    bound_l = obj_range - dist_thresh
    bound_u = obj_range + dist_thresh

  # Create dict with necessary values from 3D point cloud
  # 0 | x_snap = x 
  # 1 | z_snap = z
  # 2 | x
  # 3 | z
  # 4 | vx
  # 5 | vz
  # 6 | rcs
  # 7 | dts (delta ts)
  snap_channel_mapping = {'x':2, 'z':3, 'vx_comp':4, 'vz_comp':5, 'rcs':6, 'dts':7}

  # Create input image tensor [] 
  snap = np.zeros((len(snap_channels), output_size, output_size))

  if pc_3d.shape[1] == 0:
    # If there is no radar point at all, return zero valued snap
    return snap, [bound_l, bound_u], 0, np.zeros((8,))  # return 0 for empty snap

  # Init points in frustum
  # Indices for: x_snap, z_snap, x, z, vx, vz, rcs, dts (at first x_snap = x (same with z))
  frustum_points = pc_3d[[0,2,0,2,8,9,5,18],:].copy()

  # Compute 3D bounding box corners in BEV (planar)
  bbox_corners = compute_box_3d(obj['dim'], obj['location'], \
                                      obj['rotation_y'])
  bbox_corners = bbox_corners[0:4][:,[0,2]]

  # Calc frustum corners with 3D BB
  frustum_corners = calc_frustum_corners(dist_thresh, obj_range, bbox_corners, opt)

  # Calc Frustum expansion in pixels
  expansion_points = np.zeros((2, 3))
  expansion_points[:,[0,2]] = frustum_corners.copy()[0:2,:] # bottom left and bottom right
  expansion_points[:,0] += np.array([-1, 1])*opt.pillar_dims[1] # expand bounds to left and right
  expansion_bounds = project_to_image(expansion_points, calib)[:,0] # get x coordinates as expanded bbox bounds

  expansion_bounds = np.clip(expansion_bounds, 0, 1599)

  bbox_expansion = 0
  bbox_int = np.array([np.floor(expansion_bounds[0]), 
                       np.floor(bbox[1]), 
                       np.ceil(expansion_bounds[1]), 
                       np.ceil(bbox[3])], np.int32) # format: xyxy | get conservative discretized estimate of bbox

  roi_indices = pc_snap_proj[0, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1] # Region of Interest corresponding to 2D detectio
  # Get unique indices
  roi_indices = np.unique(roi_indices)
  # Get rid of -1 index (from init)
  roi_indices = roi_indices[roi_indices>0].astype(np.uint32)

  if roi_indices.shape[0] == 0:
    # If there is no radar point in initial ROI, return zero valued snap
    return snap, [bound_l, bound_u], 0, np.zeros((8,))  # return 0 for empty snap

  # Get frustum_points in ROI
  frustum_points = frustum_points[:,roi_indices]

  ## Check whether points are inside distance bounds

  if opt.use_dist_for_frustum: # Use distance for selection
    # Check for lower distance bound
    frustum_maskl = np.where(np.linalg.norm(frustum_points[0:2,:], axis = 0) > bound_l, 1, 0)
    # Check for upper distance bound and include current frustum_mask
    frustum_masku = np.where(np.linalg.norm(frustum_points[0:2,:], axis = 0) < bound_u, 1, 0)

  else: # Use depth for selection
    # Check for lower distance bound
    frustum_maskl = np.where(frustum_points[1,:] > bound_l, 1, 0)
    frustum_masku = np.where(frustum_points[1,:] < bound_u, 1, 0)

  frustum_mask = frustum_maskl*frustum_masku

  ## Apply mask to frustum points
  frustum_points = frustum_points[:, np.nonzero(frustum_mask)[0]]
  nr_frustum_points = frustum_points.shape[1]
  if nr_frustum_points <= opt.limit_frustum_points:
    # If there are no points in frustum, return zero valued snap
    if (opt.limit_use_closest or opt.limit_use_vel) and nr_frustum_points > 0:
     return snap, [bound_l,bound_u], nr_frustum_points, frustum_points[:,np.argsort(frustum_points, axis=1)[3,0]].flatten() # return closest point
    else:
      return snap, [bound_l,bound_u], nr_frustum_points, np.zeros((8,))
  elif check_only: # only check if there is a point inside the frustum
    return None, None, nr_frustum_points, np.zeros((8,))


  if opt.use_dist_for_frustum:
    ### Transform frustum_points (x_snap, z_snap) and frustum_corners onto x-axis of camera frame (rotation)

    # Calculate angle theta from x-axis to obj
    # theta = (obj["alpha"]-obj["rotation_y"]+np.pi/2)%(2*np.pi)

    # Calculate angle to center of frustum
    theta = -(np.arctan2(frustum_corners[1,0],frustum_corners[1,1]) +\
              np.arctan2(frustum_corners[0,0],frustum_corners[0,1]))/2 + np.pi/2

    # Create rotation matrix 
    R = get_2d_rotation_matrix(-theta)
    # Rotate frustum_points
    frustum_points[0:2,:] = (R @ frustum_points[0:2,:])
    # Rotate frustum corners
    frustum_corners = (R @ frustum_corners.T).T

    ## Calc snap_corners from frustum
    # mean in x and z
    frustum_x_min = np.amin(frustum_corners[:,0])
    frustum_z_min = np.amin(frustum_corners[:,1])

    # Distance case:
    #     since frustum is defined by dist it is circle-shaped into x-direction, 
    #     thus the amax of the corners_x is always too small than the dist.
    # Depth case:
    #     We have to guarantee that the frustum corners are within the snap 
    frustum_x_max = np.amax(np.concatenate((np.array([bound_u]), frustum_corners[:,0])))
    frustum_z_max = np.amax(frustum_corners[:,1])

    # Snap center in camera coodinate frame
    snap_center = np.array([frustum_x_min+(frustum_x_max-frustum_x_min)/2, frustum_z_min+(frustum_z_max-frustum_z_min)/2])
    # Size of untransformed snap size
    snap_size = np.amax(np.array((np.abs(frustum_x_max-frustum_x_min), np.abs(frustum_z_max-frustum_z_min))))

    # Frustum trafo from bottom left frustum corner to origin
    frustum_trafo = (snap_center-snap_size/2)

    ### Transform (translation) frustum_points (x_snap, z_snap) from snap_corners 
    # (bottom left) to camera frame
    frustum_points[0:2,:] -= np.array([frustum_trafo]).T


  else:
    # Use depth for frustum

    ## Calc snap_corners from frustum
    # mean in x and z
    frustum_x_min = np.amin(frustum_corners[:,0])
    frustum_z_min = np.amin(frustum_corners[:,1])

    # Distance case:
    #     since frustum is defined by dist it is circle-shaped into x-direction, 
    #     thus the amax of the corners_x is always too small than the dist.
    # Depth case:
    #     We have to guarantee that the frustum corners are within the snap 
    frustum_x_max = np.amax(frustum_corners[:,0])
    frustum_z_max = np.amax(frustum_corners[:,1])

    # Snap center in camera coodinate frame
    snap_center = np.array([frustum_x_min+(frustum_x_max-frustum_x_min)/2, frustum_z_min+(frustum_z_max-frustum_z_min)/2])
    # Size of untransformed snap size
    snap_size = np.amax(np.array((np.abs(frustum_x_max-frustum_x_min), np.abs(frustum_z_max-frustum_z_min))))

    frustum_trafo = (snap_center-snap_size/2)

    ### Transform (translation) frustum_points (x_snap, z_snap) from snap_corners 
    # (bottom left) to camera frame
    frustum_points[0:2,:] -= np.array([frustum_trafo]).T

  ### Scale frustum_points (x_snap, z_snap) to LFANet input (integer) using the 
  # size of the snap
  scale_2Dtosnap = (output_size-1)/snap_size
  frustum_points[0:2,:] = frustum_points[0:2,:]*scale_2Dtosnap

  # Sort frustum points from biggest z to smallest z (transformed snap coordinates)
  frustum_sorted_indices = (np.argsort(frustum_points[3,:])[::-1]).copy()

  ### Store frustum point information in snap
  # Loop over all points in frustum
  for id_fp in frustum_sorted_indices:
    # Calculate lfa_pillar_size
    if opt.lfa_pillar_size > 0 or opt.lfa_pillar_pixel > 0:
      # Create pillars out of the frustum points
      pillar_center = np.array(frustum_points[0:2, id_fp],dtype=np.int)

      # Calculate pillar pixel width. The value is rounded and therefore produces
      # rather too big than too small pillars
      if opt.lfa_pillar_pixel == 0:
        pillar_pixel_w =(opt.lfa_pillar_size/2*scale_2Dtosnap).round()
      else:
        pillar_pixel_w = (opt.lfa_pillar_pixel-1)/2

      # Calculate upper left corner of pillar
      pillar_min = pillar_center - pillar_pixel_w
      # Calculate bottom right corner of pillar
      pillar_max = pillar_center + pillar_pixel_w

      # Calculate pillar indices in x to fill snap
      snap_indices_x = np.arange(pillar_min[0],pillar_max[0]+1)
      # Make sure to not exceed the boundaries of the snap
      snap_indices_x = np.clip(snap_indices_x, 0, output_size-1)
      # Repeat the indices to get all indices in the pillar
      snap_indices_x = np.repeat(snap_indices_x,snap_indices_x.shape[0]).astype(np.int)
      # Calculate pillar indices in y to fill snap
      snap_indices_y = np.arange(pillar_min[1],pillar_max[1]+1)
      # Make sure to not exceed the boundaries of the snap
      snap_indices_y = np.clip(snap_indices_y, 0, output_size-1)
      # Repeat in opposing dimension and reshape to tensor of dim [N]
      snap_indices_y = np.tile(snap_indices_y, snap_indices_y.shape[0]).astype(np.int)

    else:
      snap_indices_x = int(frustum_points[0, id_fp])
      snap_indices_y = int(frustum_points[1, id_fp])
        
    # Loop over all channels
    for id_channel, channel in enumerate(snap_channels):
      if channel == 'cat_id':
        # Normalize category
        if opt.normalize_cat_input_lfa:
          # Normalize categories by setting mean to zero
          snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id'] - opt.cat_norm
        else:
          snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id']
      elif channel == 'cat':
        snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id'] - 1 # ann contains id not index
      else:
        frustum_row = snap_channel_mapping[channel]
        # Assign value to correct channel in snap 
        snap[id_channel, snap_indices_x, snap_indices_y] = frustum_points[frustum_row, id_fp]

  # Normalize depth
  if opt.normalize_depth and opt.normalize_depth_input_lfa and 'z' in opt.snap_channels:
      # Normalize depth values by dividing through maximal depth possible
      snap[opt.lfa_index_z,:,:] /= opt.max_pc_depth

  # Normalize time stamp
  if 'dts' in opt.snap_channels:
      # Normalize time steps by dividing through maximal time step possible
      snap[opt.lfa_index_dts,:,:] /= opt.dts_norm

  # Check that snap is computed correctly
  # LFANet only works with square resolutions
  assert snap.shape[0] == opt.lfa_channel_in, f"Wrong nr of input channels. Expected {opt.lfa_channel_in}."
  assert snap.shape[1] == opt.snap_resolution, "Wrong snap dimension. See docs."
  assert snap.shape[2] == opt.snap_resolution, "Wrong snap dimension. Dimension has to be square. See docs."

  return snap, [bound_l,bound_u], nr_frustum_points, np.zeros((8,)) 



def generate_snap_proj_torch(pc_3d, pc_snap_proj, obj, bbox, dist_thresh, trans_original, opt, calib, check_only=False):
  """pc_3d, obj, dist_thresh, self.opt
  This function generates the input image for the LFANet.

  
  :param pc_hm: Heatmap with the selected features (from one point) as channels. Only given as argument to be "returned"
  :param pc_3d: tensor 3D pointcloud of the radar including all 18 images (filtered for points above distance threshold (see opts.py))    
  :param obj: Dict containing information of the obj regarded. Contains fields:
                  "location": Center of bounding box in 3D space [x,y,z] in camera CS
                  "rotation": Global orientation of obj around Y-axis of camera
                  "dimension": Dimension in X,Y and Z of 3D-BB
                  [optional] "cat": Category/class of obj
  :param dist_thresh: Distance threshold for frustum association calculated from 3D bounding box corner.
                      The dimensions of this param change whether opt.use_dist_for_frustum is True or not.
                      If False: dist_thresh is scalar, if True: dist_thresh is [2,] tensor describing lower and upper bound
  :param opt: dict Options
  :param check_only: Flag to save computational effort when training with gt heatmap. If True, return 1 
                     when there is a radar point inside the frustum and skip computing the actual snap!
                     Default: False
  
  :return frustum_points: Return the frustum_points for filling in the features of the closest point (or any point) into pc_bb_hm when there are not enough point in the frustum for the LFANet to work with. It is None if either the snap is totally empty or has enough points for LFANet.
  """

  device = pc_3d.device
  snap_channels = opt.snap_channels
  output_size = opt.snap_resolution
  
  bbox = bbox.clone().reshape(2,2)

  bbox = bbox@trans_original[0:2,0:2]
  bbox[0,:] += trans_original[:,2].T
  bbox[1,:] += trans_original[:,2].T
  bbox = bbox.flatten()

  if opt.use_dist_for_frustum:
    # # Rotation approach.
    bound_l = dist_thresh[0]
    bound_u = dist_thresh[1]
    obj_range = torch.linalg.norm(torch.tensor((obj['location'][0],obj['location'][2]), device=device))

  else:
    # Approach without rotation to maximize size of frustum in snap
    obj_range = obj['location'][2]
    bound_l = obj_range - dist_thresh
    bound_u = obj_range + dist_thresh

  # Create dict with necessary values from 3D point cloud
  # 0 | x_snap = x 
  # 1 | z_snap = z
  # 2 | x
  # 3 | z
  # 4 | vx
  # 5 | vz
  # 6 | rcs
  # 7 | dts (delta ts)
  snap_channel_mapping = {'x':2, 'z':3, 'vx_comp':4, 'vz_comp':5, 'rcs':6, 'dts':7}

  # Create input image tensor [] 
  snap = torch.zeros((len(snap_channels), output_size, output_size), device=device)

  # Filter empty radar points (if z==0, the point is not used)
  if pc_3d.shape[1] == 0:
    # If there is no radar point at all, return zero valued snap
    return snap, [bound_l,bound_u], 0, torch.zeros((8,)), None  # return 0 for empty snap

  # Init points in frustum
  # Indices for: x_snap, z_snap, x, z, vx, vz, rcs, dts (at first x_snap = x (same with z))
  frustum_points = pc_3d[[0,2,0,2,8,9,5,18],:].clone().type(torch.float32)

  # Compute 3D bounding box corners in BEV (planar)
  bbox_corners = compute_box_3d_torch(obj['dim'], obj['location'], \
                                      obj['rotation_y'], device=device)
  bbox_corners = bbox_corners[0:4][:,[0,2]]

  # Calc frustum corners with 3D BB
  frustum_corners = calc_frustum_corners_torch(dist_thresh, obj_range, bbox_corners, opt)

  # Calc Frustum expansion in pixels
  expansion_points = torch.zeros((3,2), dtype=torch.float32, device=device)
  expansion_points[[0,2],:] = frustum_corners.clone()[0:2,:].T.type(torch.float32) # bottom left and bottom right
  expansion_points[0,:] += torch.tensor([-1, 1],device=device)*opt.pillar_dims[1] # expand bounds to left and right
  expansion_bounds = project_to_image_torch(expansion_points, calib)[0,:] # get x coordinates as expanded bbox bounds

  expansion_bounds = torch.clamp(expansion_bounds, 0, 1599)

  ### Check whether points are part of frustum
  bbox_int = torch.tensor([torch.floor(expansion_bounds[0]), 
                           torch.floor(bbox[1]), 
                           torch.ceil(expansion_bounds[1]), 
                           torch.ceil(bbox[3])]).type(torch.long) # format: xyxy | get conservative discretized estimate of bbox

  roi_indices = pc_snap_proj[0, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1] # Region of Interest corresponding to 2D detectio
  # Get unique indices
  roi_indices = torch.unique(roi_indices)
  # Get rid of -1 index (from init)
  roi_indices = roi_indices[roi_indices>0].type(torch.long)

  if roi_indices.shape[0] == 0:
    # If there is no radar point in initial ROI, return zero valued snap
    return snap, [bound_l, bound_u], 0, torch.zeros((8,)), None  # return 0 for empty snap

  # Get frustum_points in ROI
  frustum_points = frustum_points[:,roi_indices]

  ## Check whether points are inside distance bounds
  if opt.use_dist_for_frustum: # Use distance for selection
    # Check for lower distance bound
    frustum_maskl = torch.where(torch.linalg.norm(frustum_points[0:2,:], dim = 0) > bound_l, 1, 0)
    frustum_masku = torch.where(torch.linalg.norm(frustum_points[0:2,:], dim = 0) < bound_u, 1, 0)

  else: # Use depth for selection
    # Check for lower distance bound
    frustum_maskl = torch.where(frustum_points[1,:] > bound_l, 1, 0)
    frustum_masku = torch.where(frustum_points[1,:] < bound_u, 1, 0)

  frustum_mask = frustum_maskl*frustum_masku

  ## Apply mask to frustum points
  frustum_points = frustum_points[:, torch.nonzero(frustum_mask).squeeze(1)]
  nr_frustum_points = frustum_points.shape[1]
  if nr_frustum_points <= opt.limit_frustum_points:
    # If there are no points in frustum, return zero valued snap
    # if opt.debug > 0:
    #   print('Found NO points.')
    if (opt.limit_use_closest or opt.limit_use_vel) and nr_frustum_points > 0:
      return snap, [bound_l,bound_u], nr_frustum_points, frustum_points[:,torch.argsort(frustum_points, axis=1)[3,0]].flatten(), None # return closest point 
    else:
      return snap, [bound_l,bound_u], nr_frustum_points, torch.zeros((8,)), None
  elif check_only: # only check if there is a point inside the frustum
    return None, None, nr_frustum_points, torch.zeros((8,)), None

  if opt.use_dist_for_frustum:
    ### Transform frustum_points (x_snap, z_snap) and frustum_corners onto x-axis of camera frame (rotation)

    # Calculate angle to center of frustum
    theta = -(torch.atan2(frustum_corners[1,0],frustum_corners[1,1]) +\
              torch.atan2(frustum_corners[0,0],frustum_corners[0,1]))/2 + np.pi/2

    # Create rotation matrix 
    R = get_2d_rotation_matrix_torch(-theta, device)
    # Rotate frustum_points
    frustum_points[0:2,:] = (R @ frustum_points[0:2,:])
    # Rotate frustum corners
    frustum_corners = (R @ frustum_corners.T).T


    ## Calc snap_corners from frustum
    # mean in x and z
    frustum_x_min = torch.amin(frustum_corners[:,0])
    frustum_z_min = torch.amin(frustum_corners[:,1])

    # Distance case:
    #     since frustum is defined by dist it is circle-shaped into x-direction, 
    #     thus the amax of the corners_x is always too small than the dist.
    # Depth case:
    #     We have to guarantee that the frustum corners are within the snap 
    frustum_x_max = torch.amax(torch.cat((torch.tensor([bound_u],device=device), frustum_corners[:,0])))
    frustum_z_max = torch.amax(frustum_corners[:,1])

    # Snap center in camera coodinate frame
    snap_center = torch.tensor([frustum_x_min+(frustum_x_max-frustum_x_min)/2, frustum_z_min+(frustum_z_max-frustum_z_min)/2], device=device)
    # Size of untransformed snap size
    snap_size = torch.amax(torch.tensor((torch.abs(frustum_x_max-frustum_x_min), torch.abs(frustum_z_max-frustum_z_min)), device=device))

    # Frustum trafo from bottom left frustum corner to origin
    frustum_trafo = (snap_center-snap_size/2)

    ### Transform (translation) frustum_points (x_snap, z_snap) from snap_corners 
    # (bottom left) to camera frame
    frustum_points[0:2,:] -= frustum_trafo.unsqueeze(1)


  else:
    # Use depth for frustum

    ## Calc snap_corners from frustum
    # mean in x and z
    frustum_x_min = torch.amin(frustum_corners[:,0])
    frustum_z_min = torch.amin(frustum_corners[:,1])

    # Distance case:
    #     since frustum is defined by dist it is circle-shaped into x-direction, 
    #     thus the amax of the corners_x is always too small than the dist.
    # Depth case:
    #     We have to guarantee that the frustum corners are within the snap 
    frustum_x_max = torch.amax(frustum_corners[:,0])
    frustum_z_max = torch.amax(frustum_corners[:,1])

    # Snap center in camera coodinate frame
    snap_center = torch.tensor([frustum_x_min+(frustum_x_max-frustum_x_min)/2, frustum_z_min+(frustum_z_max-frustum_z_min)/2], device=device)
    # Size of untransformed snap size
    snap_size = torch.amax(torch.tensor((torch.abs(frustum_x_max-frustum_x_min), torch.abs(frustum_z_max-frustum_z_min)), device=device))

    frustum_trafo = (snap_center-snap_size/2)

    ### Transform (translation) frustum_points (x_snap, z_snap) from snap_corners 
    # (bottom left) to camera frame
    frustum_points[0:2,:] -= frustum_trafo.unsqueeze(dim=1)

  ### Scale frustum_points (x_snap, z_snap) to LFANet input (integer) using the 
  # size of the snap
  scale_2Dtosnap = (output_size-1)/snap_size
  frustum_points[0:2,:] = frustum_points[0:2,:]*scale_2Dtosnap

  # Sort frustum points from biggest z to smallest z
  _, frustum_sorted_indices = torch.sort(frustum_points[3,:], descending=True)


  ### Store frustum point information in snap
  # Loop over all points in frustum
  for id_fp in frustum_sorted_indices:
    # If wanted, calculate lfa_pillar_size
    if opt.lfa_pillar_size > 0 or opt.lfa_pillar_pixel > 0:
      # Create pillars out of the frustum points
      pillar_center = frustum_points[0:2, id_fp].type(torch.int)

      # Calculate pillar pixel width. The value is rounded and therefore produces
      # rather too big than too small pillars
      if opt.lfa_pillar_pixel == 0:
        pillar_pixel_w =(opt.lfa_pillar_size/2*scale_2Dtosnap).round()
      else:
        pillar_pixel_w = (opt.lfa_pillar_pixel-1)/2

      # Calculate upper left corner of pillar
      pillar_min = pillar_center - pillar_pixel_w
      # Calculate bottom right corner of pillar
      pillar_max = pillar_center + pillar_pixel_w

      # Calculate pillar indices in x to fill snap
      snap_indices_x = torch.arange(pillar_min[0],pillar_max[0]+1)
      # Make sure to not exceed the boundaries of the snap
      snap_indices_x = torch.clamp(snap_indices_x, min=0, max=output_size-1)
      # Repeat the indices to get all indices in the pillar
      snap_indices_x = torch.repeat_interleave(snap_indices_x, snap_indices_x.shape[0]).type(torch.long)
      # snap_indices_x = torch.reshape(snap_indices_x.unsqueeze(1).repeat(1,snap_indices_x.shape[0]), (-1,)).type(torch.long)
      # Calculate pillar indices in y to fill snap
      snap_indices_y = torch.arange(pillar_min[1],pillar_max[1]+1)
      # Make sure to not exceed the boundaries of the snap
      snap_indices_y = torch.clamp(snap_indices_y, min=0, max=output_size-1)
      # Repeat in opposing dimension and reshape to tensor of dim [N]
      snap_indices_y = torch.tile(snap_indices_y, (1,snap_indices_y.shape[0])).squeeze().type(torch.long)

    else:
      snap_indices_x = int(frustum_points[0, id_fp])
      snap_indices_y = int(frustum_points[1, id_fp])
        
    # Loop over all channels
    for id_channel, channel in enumerate(snap_channels):
      if channel == 'cat_id':
        # Normalize category
        if opt.normalize_cat_input_lfa:
          # Normalize categories by setting mean to zero
          snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id'].to(dtype=torch.float32) - opt.cat_norm
        else:
          snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id'].to(dtype=torch.float32)
      elif channel == 'cat':
        snap[id_channel, snap_indices_x, snap_indices_y] = obj['cat'].to(dtype=torch.float32)
      else:
        frustum_row = snap_channel_mapping[channel]
        # Assign value to correct channel in snap 
        snap[id_channel, snap_indices_x, snap_indices_y] = frustum_points[frustum_row, id_fp]

  # Normalize channels if required

  if opt.normalize_depth and opt.normalize_depth_input_lfa and 'z' in opt.snap_channels:
    # Normalize depth values by dividing through self.opt.max_radar_distance
    snap[opt.lfa_index_z,:,:] /= opt.max_pc_depth

  # Normalize time stamp
  if 'dts' in opt.snap_channels:
    # Normalize time stamp values by dividing through self.opt.dts_norm
    snap[opt.lfa_index_dts,:,:] /= opt.dts_norm

  # Check that snap is computed correctly
  # LFANet only works with square resolutions
  assert snap.shape[0] == opt.lfa_channel_in, f"Wrong nr of input channels. Expected {opt.lfa_channel_in}."
  assert snap.shape[1] == opt.snap_resolution, "Wrong snap dimension. See docs."
  assert snap.shape[2] == opt.snap_resolution, "Wrong snap dimension. Dimension has to be square. See docs."

  return snap, [bound_l,bound_u], nr_frustum_points, torch.zeros((8,)), frustum_points


##############################################################################################################################
############################################################## BEV ###########################################################
##############################################################################################################################

def generate_snap_BEV(pc_3d, obj, dist_thresh, opt, check_only=False):
  """pc_3d, obj, dist_thresh, self.opt
  This function generates the input image for the LFANet.

  
  :param pc_hm: Heatmap with the selected features (from one point) as channels. Only given as argument to be "returned"
  :param pc_3d: tensor 3D pointcloud of the radar including all 18 images (filtered for points above distance threshold (see opts.py))    
  :param obj: Dict containing information of the obj regarded. Contains fields:
                  "location": Center of bounding box in 3D space [x,y,z] in camera CS
                  "rotation": Global orientation of obj around Y-axis of camera
                  "dimension": Dimension in X,Y and Z of 3D-BB
                  [optional] "cat": Category/class of obj
  :param dist_thresh: Distance threshold for frustum association calculated from 3D bounding box corner.
                      The dimensions of this param change whether opt.use_dist_for_frustum is True or not.
                      If False: dist_thresh is scalar, if True: dist_thresh is [2,] tensor describing lower and upper bound
  :param opt: dict Options
  :param check_only: Flag to save computational effort when training with gt heatmap. If True, return 1 
                     when there is a radar point inside the frustum and skip computing the actual snap!
                     Default: False
  
  :return frustum_points: Return the frustum_points for filling in the features of the closest point (or any point) into pc_bb_hm when there are not enough point in the frustum for the LFANet to work with. It is None if either the snap is totally empty or has enough points for LFANet.
  """

  snap_channels = opt.snap_channels
  output_size = opt.snap_resolution

  if opt.use_dist_for_frustum:
    # # Rotation approach.

    # Use distance as range
    obj_range = np.linalg.norm(np.array((obj['location'][0],obj['location'][2])))
  else:
    # Approach without rotation to maximize size of frustum in snap
    obj_range = obj['location'][2]

  # Compute 3D bounding box corners in BEV (planar)
  bbox_corners = compute_box_3d(obj['dim'], obj['location'], \
                                      obj['rotation_y'])
  bbox_corners = bbox_corners[0:4][:,[0,2]]

  # Calc frustum corners with 3D BB
  frustum_corners = calc_frustum_corners(dist_thresh, obj_range, bbox_corners, opt)

  bound_l = np.amin(frustum_corners[:,1])
  bound_u = np.amax(frustum_corners[:,1])

  # Create dict with necessary values from 3D point cloud
  # 0 | x_snap = x 
  # 1 | z_snap = z
  # 2 | x
  # 3 | z
  # 4 | vx
  # 5 | vz
  # 6 | rcs
  # 7 | dts (delta ts)
  snap_channel_mapping = {'x':2, 'z':3, 'vx_comp':4, 'vz_comp':5, 'rcs':6, 'dts':7}

  # Create input image tensor [] 
  snap = np.zeros((len(snap_channels), output_size, output_size))

  # Filter empty radar points (if z==0, the point is not used)
  # nonzero_inds = np.nonzero(pc_3d[2,:])

  if pc_3d.shape[1] == 0:
    # If there is no radar point at all, return zero valued snap
    return snap, [bound_l, bound_u], 0, np.zeros((8,))  # return 0 for empty snap

  # Init points in frustum
  # Indices for: x_snap, z_snap, x, z, vx, vz, rcs, dts (at first x_snap = x (same with z))
  frustum_points = pc_3d[[0,2,0,2,8,9,5,18],:].copy()

  ### Check whether points are part of frustum
  
  ## Check whether points are inside distance bounds

  if opt.use_dist_for_frustum: # Use distance for selection
    # Check for lower distance bound
    frustum_maskl = np.where(np.linalg.norm(frustum_points[0:2,:], axis = 0) > bound_l, 1, 0)
    # Check for upper distance bound and include current frustum_mask
    frustum_masku = np.where(np.linalg.norm(frustum_points[0:2,:], axis = 0) < bound_u, 1, 0)

  else: # Use depth for selection
    # Check for lower distance bound
    frustum_maskl = np.where(frustum_points[1,:] > bound_l, 1, 0)
    frustum_masku = np.where(frustum_points[1,:] < bound_u, 1, 0)

  frustum_mask = frustum_maskl*frustum_masku


  # Check whether points are between both straight edges defined by the angles
  # Output is points in cone (in intersection with ring of points is the frustum)
  
  # Compute cost of half-spaces
  # Iterate over all frustum points that are not masked yet
  non_zero_mask = np.nonzero(frustum_mask)[0]
  for i in non_zero_mask:
    inside = 1
    # Iterate over the first 2 edges of the frustum (bottom angle min and bottom angle max)
    for edge in range(2): # point corresponding to smallest (0) and biggest (1) angle
      x1 = 0
      z1 = 0
      x2 = frustum_corners[edge,0] 
      z2 = frustum_corners[edge,1] 

      if edge == 0: # for orientation of edge
          cache1 = x1
          cache2 = z1
          z1 = z2
          x1 = x2
          x2 = cache1
          z2 = cache2

      # Represent edge by a line
      a0 = - ( z1 - z2 )  # compute a0 for current edge
      a1 = - ( x2 - x1 )  # compute a1 for current edge
      b = - ( z1*x2 - z2*x1 )
      # Evaluate half-space of each edge at radar point
      half_space_eq = b - a0*frustum_points[2,i] - a1*frustum_points[3,i]
      # If point is outside w.r.t. one half space set inside to zero
      inside *= max(0.0, half_space_eq)
    # Store information is point is inside or not
    frustum_mask[i] = 1 if inside > 0 else 0  

  ## Apply mask to frustum points
  frustum_points = frustum_points[:, np.nonzero(frustum_mask)[0]]
  nr_frustum_points = frustum_points.shape[1]
  if nr_frustum_points <= opt.limit_frustum_points:
    # If there are no points in frustum, return zero valued snap
    # if opt.debug > 0:
    #   print('Found NO points.')
    if (opt.limit_use_closest or opt.limit_use_vel) and nr_frustum_points > 0:
      return snap, [bound_l,bound_u], nr_frustum_points, frustum_points[:,np.argsort(frustum_points, axis=1)[3,0]].flatten() # return closest point
    else:
      return snap, [bound_l,bound_u], nr_frustum_points, np.zeros((8,))
  elif check_only: # only check if there is a point inside the frustum
    return None, None, nr_frustum_points, np.zeros((8,))

  if opt.use_dist_for_frustum:
    ### Transform frustum_points (x_snap, z_snap) and frustum_corners onto x-axis of camera frame (rotation)

    # Calculate angle theta from x-axis to obj
    # theta = (obj["alpha"]-obj["rotation_y"]+np.pi/2)%(2*np.pi)

    # Calculate angle to center of frustum
    theta = -(np.arctan2(frustum_corners[1,0],frustum_corners[1,1]) +\
              np.arctan2(frustum_corners[0,0],frustum_corners[0,1]))/2 + np.pi/2

    # Create rotation matrix 
    R = get_2d_rotation_matrix(-theta)
    # Rotate frustum_points
    frustum_points[0:2,:] = (R @ frustum_points[0:2,:])
    # Rotate frustum corners
    frustum_corners = (R @ frustum_corners.T).T

    ## Calc snap_corners from frustum
    # mean in x and z
    frustum_x_min = np.amin(frustum_corners[:,0])
    frustum_z_min = np.amin(frustum_corners[:,1])

    # Distance case:
    #     since frustum is defined by dist it is circle-shaped into x-direction, 
    #     thus the amax of the corners_x is always too small than the dist.
    # Depth case:
    #     We have to guarantee that the frustum corners are within the snap 
    frustum_x_max = np.amax(np.concatenate((np.array([bound_u]), frustum_corners[:,0])))
    frustum_z_max = np.amax(frustum_corners[:,1])

    # Snap center in camera coodinate frame
    snap_center = np.array([frustum_x_min+(frustum_x_max-frustum_x_min)/2, frustum_z_min+(frustum_z_max-frustum_z_min)/2])
    # Size of untransformed snap size
    snap_size = np.amax(np.array((np.abs(frustum_x_max-frustum_x_min), np.abs(frustum_z_max-frustum_z_min))))

    # Frustum trafo from bottom left frustum corner to origin
    frustum_trafo = (snap_center-snap_size/2)

    ### Transform (translation) frustum_points (x_snap, z_snap) from snap_corners 
    # (bottom left) to camera frame
    frustum_points[0:2,:] -= np.array([frustum_trafo]).T

  else:
    # Use depth for frustum

    ## Calc snap_corners from frustum
    # mean in x and z
    frustum_x_min = np.amin(frustum_corners[:,0])
    frustum_z_min = np.amin(frustum_corners[:,1])

    # Distance case:
    #     since frustum is defined by dist it is circle-shaped into x-direction, 
    #     thus the amax of the corners_x is always too small than the dist.
    # Depth case:
    #     We have to guarantee that the frustum corners are within the snap 
    frustum_x_max = np.amax(frustum_corners[:,0])
    frustum_z_max = np.amax(frustum_corners[:,1])

    # Snap center in camera coodinate frame
    snap_center = np.array([frustum_x_min+(frustum_x_max-frustum_x_min)/2, frustum_z_min+(frustum_z_max-frustum_z_min)/2])
    # Size of untransformed snap size
    snap_size = np.amax(np.array((np.abs(frustum_x_max-frustum_x_min), np.abs(frustum_z_max-frustum_z_min))))

    frustum_trafo = (snap_center-snap_size/2)

    ### Transform (translation) frustum_points (x_snap, z_snap) from snap_corners 
    # (bottom left) to camera frame
    frustum_points[0:2,:] -= np.array([frustum_trafo]).T

  ### Scale frustum_points (x_snap, z_snap) to LFANet input (integer) using the 
  # size of the snap
  scale_2Dtosnap = (output_size-1)/snap_size
  frustum_points[0:2,:] = frustum_points[0:2,:]*scale_2Dtosnap

  # Sort frustum points from biggest z to smallest z (transformed snap coordinates)
  frustum_sorted_indices = (np.argsort(frustum_points[3,:])[::-1]).copy()

  ### Store frustum point information in snap
  # Loop over all points in frustum
  for id_fp in frustum_sorted_indices:
    # If wanted, calculate lfa_pillar_size
    if opt.lfa_pillar_size > 0 or opt.lfa_pillar_pixel > 0:
      # Create pillars out of the frustum points
      pillar_center = np.array(frustum_points[0:2, id_fp],dtype=np.int)

      # Calculate pillar pixel width. The value is rounded and therefore produces
      # rather too big than too small pillars
      if opt.lfa_pillar_pixel == 0:
        pillar_pixel_w =(opt.lfa_pillar_size/2*scale_2Dtosnap).round()
      else:
        pillar_pixel_w = (opt.lfa_pillar_pixel-1)/2

      # Calculate upper left corner of pillar
      pillar_min = pillar_center - pillar_pixel_w
      # Calculate bottom right corner of pillar
      pillar_max = pillar_center + pillar_pixel_w

      # Calculate pillar indices in x to fill snap
      snap_indices_x = np.arange(pillar_min[0],pillar_max[0]+1)
      # Make sure to not exceed the boundaries of the snap
      snap_indices_x = np.clip(snap_indices_x, 0, output_size-1)
      # Repeat the indices to get all indices in the pillar
      snap_indices_x = np.repeat(snap_indices_x,snap_indices_x.shape[0]).astype(np.int)
      # Calculate pillar indices in y to fill snap
      snap_indices_y = np.arange(pillar_min[1],pillar_max[1]+1)
      # Make sure to not exceed the boundaries of the snap
      snap_indices_y = np.clip(snap_indices_y, 0, output_size-1)
      # Repeat in opposing dimension and reshape to tensor of dim [N]
      snap_indices_y = np.tile(snap_indices_y, snap_indices_y.shape[0]).astype(np.int)

    else:
      snap_indices_x = int(frustum_points[0, id_fp])
      snap_indices_y = int(frustum_points[1, id_fp])
        
    # Loop over all channels
    for id_channel, channel in enumerate(snap_channels):
      if channel == 'cat_id':
        # Normalize category
        if opt.normalize_cat_input_lfa:
          # Normalize categories by setting mean to zero
          snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id'] - opt.cat_norm
        else:
          snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id']
      elif channel == 'cat':
        snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id'] - 1 # ann contains id not index
      else:
        frustum_row = snap_channel_mapping[channel]
        # Assign value to correct channel in snap 
        snap[id_channel, snap_indices_x, snap_indices_y] = frustum_points[frustum_row, id_fp]

  if opt.normalize_depth and opt.normalize_depth_input_lfa and 'z' in opt.snap_channels:
      # Normalize depth values by dividing through self.opt.max_radar_distance
      snap[opt.lfa_index_z,:,:] /= opt.max_pc_depth

  # Normalize time stamp
  if 'dts' in opt.snap_channels:
      # Normalize depth values by dividing through self.opt.max_radar_distance
      snap[opt.lfa_index_dts,:,:] /= opt.dts_norm

  # Check that snap is computed correctly
  # LFANet only works with square resolutions
  assert snap.shape[0] == opt.lfa_channel_in, f"Wrong nr of input channels. Expected {opt.lfa_channel_in}."
  assert snap.shape[1] == opt.snap_resolution, "Wrong snap dimension. See docs."
  assert snap.shape[2] == opt.snap_resolution, "Wrong snap dimension. Dimension has to be square. See docs."

  return snap, [bound_l,bound_u], nr_frustum_points, np.zeros((8,)) 


def generate_snap_BEV_torch(pc_3d, obj, dist_thresh, opt, check_only=False):
  """pc_3d, obj, dist_thresh, self.opt
  This function generates the input image for the LFANet.

  
  :param pc_hm: Heatmap with the selected features (from one point) as channels. Only given as argument to be "returned"
  :param pc_3d: tensor 3D pointcloud of the radar including all 18 images (filtered for points above distance threshold (see opts.py))    
  :param obj: Dict containing information of the obj regarded. Contains fields:
                  "location": Center of bounding box in 3D space [x,y,z] in camera CS
                  "rotation": Global orientation of obj around Y-axis of camera
                  "dimension": Dimension in X,Y and Z of 3D-BB
                  [optional] "cat": Category/class of obj
  :param dist_thresh: Distance threshold for frustum association calculated from 3D bounding box corner.
                      The dimensions of this param change whether opt.use_dist_for_frustum is True or not.
                      If False: dist_thresh is scalar, if True: dist_thresh is [2,] tensor describing lower and upper bound
  :param opt: dict Options
  :param check_only: Flag to save computational effort when training with gt heatmap. If True, return 1 
                     when there is a radar point inside the frustum and skip computing the actual snap!
                     Default: False
  
  :return frustum_points: Return the frustum_points for filling in the features of the closest point (or any point) into pc_bb_hm when there are not enough point in the frustum for the LFANet to work with. It is None if either the snap is totally empty or has enough points for LFANet.
  """

  device = copy.deepcopy(pc_3d.device)
  snap_channels = opt.snap_channels
  output_size = opt.snap_resolution

  if opt.use_dist_for_frustum:
    # # Rotation approach.

    # Use distance as range
    obj_range = torch.linalg.norm(torch.tensor((obj['location'][0],obj['location'][2])))\
                                    .to(device=device)  
  else:
    # Approach without rotation to maximize size of frustum in snap

    obj_range = obj['location'][2]

  # Compute 3D bounding box corners in BEV (planar)
  bbox_corners = compute_box_3d_torch(obj['dim'], obj['location'], \
                                      obj['rotation_y'], device=device)
  bbox_corners = bbox_corners[0:4][:,[0,2]]

  # Calc frustum corners with 3D BB
  frustum_corners = calc_frustum_corners_torch(dist_thresh, obj_range, bbox_corners, opt)\
                        .type(dtype=torch.float32)

  bound_l = torch.amin(frustum_corners[:,1])
  bound_u = torch.amax(frustum_corners[:,1])

  # Create dict with necessary values from 3D point cloud
  # 0 | x_snap = x 
  # 1 | z_snap = z
  # 2 | x
  # 3 | z
  # 4 | vx
  # 5 | vz
  # 6 | rcs
  # 7 | dts (delta ts)
  snap_channel_mapping = {'x':2, 'z':3, 'vx_comp':4, 'vz_comp':5, 'rcs':6, 'dts':7}

  # Create input image tensor [] 
  snap = torch.zeros((len(snap_channels), output_size, output_size), device=device)

  # Filter empty radar points (if z==0, the point is not used)
  if pc_3d.shape[1] == 0:
    # If there is no radar point at all, return zero valued snap
    return snap, [bound_l,bound_u], 0, torch.zeros((8,)), None  # return 0 for empty snap

  # Init points in frustum
  # Indices for: x_snap, z_snap, x, z, vx, vz, rcs, dts (at first x_snap = x (same with z))
  frustum_points = pc_3d.clone()[[0,2,0,2,8,9,5,18],:].type(torch.float32)

  ### Check whether points are part of frustum
  
  ## Check whether points are inside distance bounds
  if opt.use_dist_for_frustum: # Use distance for selection
    # Check for lower distance bound
    frustum_maskl = torch.where(torch.linalg.norm(frustum_points[0:2,:], dim = 0) > bound_l, 1, 0)
    frustum_masku = torch.where(torch.linalg.norm(frustum_points[0:2,:], dim = 0) < bound_u, 1, 0)

  else: # Use depth for selection
    # Check for lower distance bound
    frustum_maskl = torch.where(frustum_points[1,:] > bound_l, 1, 0)
    frustum_masku = torch.where(frustum_points[1,:] < bound_u, 1, 0)

  frustum_mask = frustum_maskl*frustum_masku

  # Check whether points are between both straight edges defined by the angles
  # Output is points in cone (in intersection with ring of points is the frustum)
  
  # Compute cost of half-spaces
  # Iterate over all frustum points that are not masked yet
  non_zero_mask = torch.nonzero(frustum_mask)
  for i in non_zero_mask:
    inside = 1
    # Iterate over the first 2 edges of the frustum (bottom angle min and bottom angle max)
    for edge in range(2): # point corresponding to smallest (0) and biggest (1) angle
      x1 = 0
      z1 = 0
      x2 = frustum_corners[edge,0] 
      z2 = frustum_corners[edge,1] 

      if edge == 0: # for orientation of edge
          cache1 = x1
          cache2 = z1
          z1 = z2
          x1 = x2
          x2 = cache1
          z2 = cache2

      # Represent edge by a line
      a0 = - ( z1 - z2 )  # compute a0 for current edge
      a1 = - ( x2 - x1 )  # compute a1 for current edge
      b = - ( z1*x2 - z2*x1 )
      # Evaluate half-space of each edge at radar point
      half_space_eq = b - a0*frustum_points[2,i] - a1*frustum_points[3,i]
      # If point is outside w.r.t. one half space set inside to zero
      inside *= max(0.0, half_space_eq)
    # Store information is point is inside or not
    frustum_mask[i] = 1 if inside > 0 else 0  

  ## Apply mask to frustum points
  frustum_points = frustum_points[:, torch.nonzero(frustum_mask).squeeze(1)]
  nr_frustum_points = frustum_points.shape[1]
  if nr_frustum_points <= opt.limit_frustum_points:
    # If there are no points in frustum, return zero valued snap
    # if opt.debug > 0:
    #   print('Found NO points.')
    if (opt.limit_use_closest or opt.limit_use_vel) and nr_frustum_points > 0:
      return snap, [bound_l,bound_u], nr_frustum_points, frustum_points[:,torch.argsort(frustum_points, axis=1)[3,0]].flatten(), None # return closest point
    else:
      return snap, [bound_l,bound_u], nr_frustum_points, torch.zeros((8,)), None
  elif check_only: # only check if there is a point inside the frustum
    return None, None, nr_frustum_points, torch.zeros((8,)), None

  if opt.use_dist_for_frustum:
    ### Transform frustum_points (x_snap, z_snap) and frustum_corners onto x-axis of camera frame (rotation)

    # Calculate angle to center of frustum
    theta = -(torch.atan2(frustum_corners[1,0],frustum_corners[1,1]) +\
              torch.atan2(frustum_corners[0,0],frustum_corners[0,1]))/2 + np.pi/2

    # Create rotation matrix 
    R = get_2d_rotation_matrix_torch(-theta, device)
    # Rotate frustum_points
    frustum_points[0:2,:] = (R @ frustum_points[0:2,:])
    # Rotate frustum corners
    frustum_corners = (R @ frustum_corners.T).T

    ## Calc snap_corners from frustum
    # mean in x and z
    frustum_x_min = torch.amin(frustum_corners[:,0])
    frustum_z_min = torch.amin(frustum_corners[:,1])

    # Distance case:
    #     since frustum is defined by dist it is circle-shaped into x-direction, 
    #     thus the amax of the corners_x is always too small than the dist.
    # Depth case:
    #     We have to guarantee that the frustum corners are within the snap 
    frustum_x_max = torch.amax(torch.cat((torch.tensor([bound_u],device=device), frustum_corners[:,0])))
    frustum_z_max = torch.amax(frustum_corners[:,1])

    # Snap center in camera coodinate frame
    snap_center = torch.tensor([frustum_x_min+(frustum_x_max-frustum_x_min)/2, frustum_z_min+(frustum_z_max-frustum_z_min)/2], device=device)
    # Size of untransformed snap size
    snap_size = torch.amax(torch.tensor((torch.abs(frustum_x_max-frustum_x_min), torch.abs(frustum_z_max-frustum_z_min)), device=device))

    # Frustum trafo from bottom left frustum corner to origin
    frustum_trafo = (snap_center-snap_size/2)

    ### Transform (translation) frustum_points (x_snap, z_snap) from snap_corners 
    # (bottom left) to camera frame
    frustum_points[0:2,:] -= frustum_trafo.unsqueeze(1)

  else:
    # Use depth for frustum

    ## Calc snap_corners from frustum
    # mean in x and z
    frustum_x_min = torch.amin(frustum_corners[:,0])
    frustum_z_min = torch.amin(frustum_corners[:,1])

    # Distance case:
    #     since frustum is defined by dist it is circle-shaped into x-direction, 
    #     thus the amax of the corners_x is always too small than the dist.
    # Depth case:
    #     We have to guarantee that the frustum corners are within the snap 
    frustum_x_max = torch.amax(frustum_corners[:,0])
    frustum_z_max = torch.amax(frustum_corners[:,1])

    # Snap center in camera coodinate frame
    snap_center = torch.tensor([frustum_x_min+(frustum_x_max-frustum_x_min)/2, frustum_z_min+(frustum_z_max-frustum_z_min)/2], device=device)
    # Size of untransformed snap size
    snap_size = torch.amax(torch.tensor((torch.abs(frustum_x_max-frustum_x_min), torch.abs(frustum_z_max-frustum_z_min)), device=device))

    frustum_trafo = (snap_center-snap_size/2)

    ### Transform (translation) frustum_points (x_snap, z_snap) from snap_corners 
    # (bottom left) to camera frame
    frustum_points[0:2,:] -= frustum_trafo.unsqueeze(dim=1)

  ### Scale frustum_points (x_snap, z_snap) to LFANet input (integer) using the 
  # size of the snap
  scale_2Dtosnap = (output_size-1)/snap_size
  frustum_points[0:2,:] = frustum_points[0:2,:]*scale_2Dtosnap

  # Sort frustum points from biggest z to smallest z
  _, frustum_sorted_indices = torch.sort(frustum_points[3,:], descending=True)

  ### Store frustum point information in snap
  # Loop over all points in frustum
  for id_fp in frustum_sorted_indices:
    # If wanted, calculate lfa_pillar_size
    if opt.lfa_pillar_size > 0 or opt.lfa_pillar_pixel > 0:
      # Create pillars out of the frustum points
      pillar_center = frustum_points[0:2, id_fp].type(torch.int)

      # Calculate pillar pixel width. The value is rounded and therefore produces
      # rather too big than too small pillars
      if opt.lfa_pillar_pixel == 0:
        pillar_pixel_w =(opt.lfa_pillar_size/2*scale_2Dtosnap).round()
      else:
        pillar_pixel_w = (opt.lfa_pillar_pixel-1)/2

      # Calculate upper left corner of pillar
      pillar_min = pillar_center - pillar_pixel_w
      # Calculate bottom right corner of pillar
      pillar_max = pillar_center + pillar_pixel_w

      # Calculate pillar indices in x to fill snap
      snap_indices_x = torch.arange(pillar_min[0],pillar_max[0]+1)
      # Make sure to not exceed the boundaries of the snap
      snap_indices_x = torch.clamp(snap_indices_x, min=0, max=output_size-1)
      # Repeat the indices to get all indices in the pillar
      snap_indices_x = torch.repeat_interleave(snap_indices_x, snap_indices_x.shape[0]).type(torch.long)
      # snap_indices_x = torch.reshape(snap_indices_x.unsqueeze(1).repeat(1,snap_indices_x.shape[0]), (-1,)).type(torch.long)
      # Calculate pillar indices in y to fill snap
      snap_indices_y = torch.arange(pillar_min[1],pillar_max[1]+1)
      # Make sure to not exceed the boundaries of the snap
      snap_indices_y = torch.clamp(snap_indices_y, min=0, max=output_size-1)
      # Repeat in opposing dimension and reshape to tensor of dim [N]
      snap_indices_y = torch.tile(snap_indices_y, (1,snap_indices_y.shape[0])).squeeze().type(torch.long)

    else:
      snap_indices_x = int(frustum_points[0, id_fp])
      snap_indices_y = int(frustum_points[1, id_fp])
        
    # Loop over all channels
    for id_channel, channel in enumerate(snap_channels):
      if channel == 'cat_id':
        # Normalize category
        if opt.normalize_cat_input_lfa:
          # Normalize categories by setting mean to zero
          snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id'].to(dtype=torch.float32) - opt.cat_norm
        else:
          snap[id_channel, snap_indices_x, snap_indices_y] = obj['category_id'].to(dtype=torch.float32)
      elif channel == 'cat':
        snap[id_channel, snap_indices_x, snap_indices_y] = obj['cat'].to(dtype=torch.float32)
      else:
        frustum_row = snap_channel_mapping[channel]
        # Assign value to correct channel in snap 
        snap[id_channel, snap_indices_x, snap_indices_y] = frustum_points[frustum_row, id_fp]

  # Normalize channels if required

  if opt.normalize_depth and opt.normalize_depth_input_lfa and 'z' in opt.snap_channels:
    # Normalize depth values by dividing through self.opt.max_radar_distance
    snap[opt.lfa_index_z,:,:] /= opt.max_pc_depth

  # Normalize time stamp
  if 'dts' in opt.snap_channels:
    # Normalize time stamp values by dividing through self.opt.dts_norm
    snap[opt.lfa_index_dts,:,:] /= opt.dts_norm

  # Check that snap is computed correctly
  # LFANet only works with square resolutions
  assert snap.shape[0] == opt.lfa_channel_in, f"Wrong nr of input channels. Expected {opt.lfa_channel_in}."
  assert snap.shape[1] == opt.snap_resolution, "Wrong snap dimension. See docs."
  assert snap.shape[2] == opt.snap_resolution, "Wrong snap dimension. Dimension has to be square. See docs."

  return snap, [bound_l,bound_u], nr_frustum_points, torch.zeros((8,)), frustum_points
