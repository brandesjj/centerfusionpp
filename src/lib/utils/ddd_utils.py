from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from asyncio import base_subprocess
import math
from scipy.spatial import ConvexHull
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from zmq import device


import matplotlib.pyplot as plt

import numpy as np
import cv2
import torch


def get_2d_rotation_matrix_torch(angle_in_radians: float, device: device) -> torch.tensor:
  """
  Makes rotation matrix to rotate point in x-z-plane counterclockwise
  by angle_in_radians.
  """

  return torch.tensor([[torch.cos(angle_in_radians), -torch.sin(angle_in_radians)],
                       [torch.sin(angle_in_radians), torch.cos(angle_in_radians)]],\
                        device=device, dtype=torch.float32)

def get_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
  """
  Makes rotation matrix to rotate point in x-z-plane counterclockwise
  by angle_in_radians.
  """

  return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                   [np.sin(angle_in_radians), np.cos(angle_in_radians)]], dtype=np.float32)



def vgt_to_vrad(v_gt:np.array, dir:np.array):
    """
    Calculates the radial velocity given the ground truth velocity.
    This function can be used to calculate the "ground truth radial velocity".
    The ground truth velocity will direct into the direction given in dir.

    :param v_gt: Ground truth velocity of the object that we currently analyze.
    :param dir: Direction of the output ground truth velocity. In case of radar point, this is the radial velocity of the point.
    :return: v_r_gt: Ground truth radial velocity projected into the direction of the radial velocity of the radar point
    """

    # Calculate radial velocity in one step
    v_r_gt = np.dot(dir, v_gt) / np.dot(dir, dir) * dir

    # Does the same but unnecessary complex
    # v_r = np.dot(radar_pos, v_gt) / (np.linalg.norm(radar_pos)*np.linalg.norm(v_gt)) * np.linalg.norm(v_gt)
    # v_r = v_r * radar_pos / np.linalg.norm(radar_pos)

    return v_r_gt

def v_to_vrad_torch(v:torch.tensor, dir:torch.tensor):
    """ Torch version 
    Calculates the radial velocity given the ground truth velocity.
    This function can be used to calculate the "ground truth radial velocity".
    The ground truth velocity will direct into the direction given in dir.

    :param v_gt: Ground truth velocity of the object that we currently analyze.
    :param dir: Direction of the output ground truth velocity. In case of radar point, this is the radial velocity of the point.
    :return: v_r_gt: Ground truth radial velocity projected into the direction of the radial velocity of the radar point
    """

    # Calculate radial velocity in one step
    v_r = torch.dot(dir, v) / torch.dot(dir, dir) * dir

    return v_r


def calc_frustum_corners(dist_thresh, 
                         ann_range,
                         bbox_corners,
                         opt):
  """
  Calc the frustum corners in the Camera CS
  
  :param dist_thresh: Distance threshold for frustum association calculated from 3D bounding box corner.
                      The dimensions of this param change whether opt.use_dist_for_frustum is True or not.
                      If False: dist_thresh is scalar, if True: dist_thresh is [2,] tensor describing lower and upper bound
  :param ann_range: Distance/Depth to centerpoint of Frustum » Distance / Depth(Nabati) to annotation
  :param bbox_corners: Corners of the 3D-bounding box in BEV, (x,z) coord in Camera CS
  :param frustum_style: Either 'depth' or 'dist' depending on the frustum one wants to draw
  :param opt: Options
  :return: frustum_corners: Corners of the Frustum. [4x2] np.ndarray with the four corners as
                          bottom inner, bottom outter, upper outter, upper inner
  """

  # If using LFANet extend frustum to match the Pillars used in CenterFusion
  # First shift the bbox_corners by pillar_size/2 to the left, then the same to the right and save all points
  if opt.use_lfa:
    bbox_corners = np.concatenate((bbox_corners, bbox_corners), 0)
    bbox_corners[0:4, 0] += - opt.pillar_dims[1]/2 # To the left
    bbox_corners[4::, 0] += opt.pillar_dims[1]/2 # To the right

  # Calculate angle to all corners from Camera CS z axis
  if not np.any(bbox_corners[:,1]==0):
    bbox_angles = np.arctan(bbox_corners[:,0]/bbox_corners[:,1])
  else:
    # Do not divide by 0
    bbox_angles = np.arctan(bbox_corners[:,0]/(bbox_corners[:,1]+1e-4))
  
  # Smallest absolut angle to the z axis of all 4 corners
  angle_in = np.amin(bbox_angles)
  # Biggest absolut angle to the z axis of all 4 corners
  angle_out = np.amax(bbox_angles)

  if opt.use_dist_for_frustum:
    # Using vector dist_thresh
    depth_b = dist_thresh[0]
    depth_u = dist_thresh[1]
  else:
    # Using scalar dist_thresh
    depth_b = ann_range - dist_thresh
    depth_u = ann_range + dist_thresh

  # angle_offset = 0

  # If using LFANet component, make sure to extend the frustum
  if opt.use_lfa:
    # # Extend angle of frustum
    # # Calculate center angle of frustum
    # angle_offset = np.arctan2(np.array(opt.pillar_dims[1]/2), depth_u)
    # # bbox_angles += torch.tensor([-1, 1, 1, -1], device=bbox_corners.device)*angle_offset
    # angle_in += - angle_offset
    # angle_out += angle_offset
  
    # Extend depth of frustum by pillar width
    depth_b += -opt.pillar_dims[1]/2
    depth_u += opt.pillar_dims[1]/2


  depth_bu = np.array([depth_b, depth_u]) # Array with depth boundaries of bottom and upper frustum bounds
  
  if opt.use_dist_for_frustum:
    # Corner with smallest and biggest angle
    # corner_in = bbox_corners[np.argmin((bbox_angles)),:]
    # corner_out = bbox_corners[np.argmax((bbox_angles)),:]
    
    dir_in = np.array([np.sin(angle_in), np.cos(angle_in)])
    dir_out = np.array([np.sin(angle_out), np.cos(angle_out)])
    direction = np.concatenate(([dir_in],[dir_out]), axis=0)
    
    # Calculate direction vectors of left and right frustum boundaries
    # dir_in = np.array(corner_in / np.linalg.norm(corner_in))
    # dir_out = np.array(corner_out / np.linalg.norm(corner_out))
    # dir = np.concatenate(([dir_in],[dir_out]), axis=0)

    # Calculate corners of the frustum in the order described above (easier to plot)
    frustum_corners = np.concatenate((direction*(depth_b), direction[::-1,:]*(depth_u)))
    
  else:
    # Use depth as estimation method (default method used by CenterFusion)
    
    frustum_corners = np.zeros((4,2))
    # z axis (camera frame)
    frustum_corners[0:2,1] = depth_b
    frustum_corners[2:4,1] = depth_u      
    # x Axis (camera frame)
    frustum_corners[[0,3],0] = np.tan(angle_in)*depth_bu
    frustum_corners[[1,2],0] = np.tan(angle_out)*depth_bu
  
  return frustum_corners


def calc_frustum_corners_torch(dist_thresh, 
                         ann_range,
                         bbox_corners,
                         opt):
  """
  Calc the frustum corners in the Camera CS
  
  :param dist_thresh: Distance threshold for frustum association calculated from 3D bounding box corner.
                      The dimensions of this param change whether opt.use_dist_for_frustum is True or not.
                      If False: dist_thresh is scalar, if True: dist_thresh is [2,] tensor describing lower and upper bound
  :param ann_range: Distance/Depth to centerpoint of Frustum » Distance / Depth(Nabati) to annotation
  :param bbox_corners: Corners of the 3D-bounding box in BEV, (x,z) coord in Camera CS
  :param frustum_style: Either 'depth' or 'dist' depending on the frustum one wants to draw
  :param opt: Options

  :return: frustum_corners: Corners of the Frustum. [4x2] np.ndarray with the four corners as
                          bottom inner, bottom outter, upper outter, upper inner

  """

  # If using LFANet extend frustum to match the Pillars used in CenterFusion
  # First shift the bbox_corners by pillar_size/2 to the left, then the same to the right and save all points
  if opt.use_lfa:
    bbox_corners = torch.cat((bbox_corners, bbox_corners), 0)
    bbox_corners[0:4, 0] += - opt.pillar_dims[1]/2 # To the left
    bbox_corners[4::, 0] += opt.pillar_dims[1]/2 # To the right

  # Calculate angle to all corners from Camera CS z axis
  bbox_angles = torch.atan(bbox_corners[:,0]/(bbox_corners[:,1]+1e-9))
  
  # Smallest absolut angle to the z axis of all 4 corners
  angle_in = torch.amin(bbox_angles)#.type(dtype=torch.float32)
  # Biggest absolut angle to the z axis of all 4 corners
  angle_out = torch.amax(bbox_angles)#.type(dtype=torch.float32)

  if opt.use_dist_for_frustum:
    # Using vector dist_thresh
    depth_b = dist_thresh[0]
    depth_u = dist_thresh[1]
  else:
    # Using scalar dist_thresh
    depth_b = ann_range - dist_thresh
    depth_u = ann_range + dist_thresh

  # If using LFANet component, make sure to extend the frustum
  if opt.use_lfa:
    # Extend depth of frustum by pillar width
    depth_b += -opt.pillar_dims[1]/2
    depth_u += opt.pillar_dims[1]/2

  # Tensor with depth boundaries of bottom and upper frustum bounds
  depth_bu = torch.tensor((depth_b, depth_u), device=bbox_corners.device) 

  if opt.use_dist_for_frustum:

    # Calculate direction vectors of left and right frustum boundaries
    dir_in = torch.tensor([torch.sin(angle_in), torch.cos(angle_in)], device=ann_range.device)
    dir_out = torch.tensor([torch.sin(angle_out), torch.cos(angle_out)], device=ann_range.device)
    direction = torch.cat((dir_in,dir_out), axis=0).view(2,2)

    # Calculate corners of the frustum in the order described above (easier to plot)
    frustum_corners = torch.cat((direction*(depth_b), direction.flip(dims=(0,))*(depth_u))).to(device=bbox_angles.device)
    
  else:

    # Use depth as estimation method (default method used by CenterFusion)
    frustum_corners = torch.zeros((4,2), device=bbox_corners.device)

    # z axis (camera frame)
    frustum_corners[0:2,1] = depth_b
    frustum_corners[2:4,1] = depth_u      

    # x Axis (camera frame)
    frustum_corners[[0,3],0] = torch.tan(angle_in)*depth_bu
    frustum_corners[[1,2],0] = torch.tan(angle_out)*depth_bu
  
  return frustum_corners

def compute_corners_3d(dim, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners).transpose(1, 0)
  return corners_3d

def compute_corners_3d_torch(dim, rotation_y, device=None):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = torch.cos(rotation_y), torch.sin(rotation_y)
    R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=torch.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = torch.tensor([x_corners, y_corners, z_corners], device=device, dtype=torch.float32)
    corners_3d = torch.mm(R, corners).transpose(1, 0)
    return corners_3d


def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  corners_3d = compute_corners_3d(dim, rotation_y)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
  return corners_3d

def compute_box_3d_torch(dim, location, rotation_y, device=None):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  corners_3d = compute_corners_3d_torch(dim, rotation_y, device)
  corners_3d = corners_3d + location.to(torch.float32).reshape(1, 3)
  return corners_3d


def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4 camera matrix 
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:] # normalize
  # import pdb; pdb.set_trace()
  return pts_2d

def project_to_image_torch(tensor_3d, P):
  """
  Project points in 3D space in tensor_3d into image plane coordinates 
  using the camera matrix P

  :param tensor_3d: torch.tensor of size [3xN] with N 3D points in camera coordinates
  :param P: camera calibration matrix P = K*[R t]
  """

  # Create homogeneous coordinates fro tensor_3d
  tensor_3d = torch.cat([tensor_3d, \
    torch.ones((1,tensor_3d.shape[1]),device=P.device)], axis=0)

  # Multiply hom. tensor with P
  pts_2d = P @ tensor_3d
  
  # Normalize pts_2d
  pts_2d *= 1 / pts_2d[2,:]

  # Check whether scaling is reasonable
  if torch.any(pts_2d[2,:]<=0):
    print("pts_3d do not lie within visible space.")

  return pts_2d[0:2,:]


def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

def draw_box_3d(image, corners, c=(255, 0, 255), same_color=False):
  face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [3,0,4,7],
              [2,3,7,6]]
  right_corners = [1, 2, 6, 5] if not same_color else []
  left_corners = [0, 3, 7, 4] if not same_color else []
  thickness = 4 if same_color else 2
  corners = corners.astype(np.int32)
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      # print('corners', corners)
      cc = c
      if (f[j] in left_corners) and (f[(j+1)%4] in left_corners):
        cc = (255, 0, 0)
      if (f[j] in right_corners) and (f[(j+1)%4] in right_corners):
        cc = (0, 0, 255)
      try:
        cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
            (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), cc, thickness, lineType=cv2.LINE_AA)
      except:
        pass
    if ind_f == 0:
      try:
        cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                 (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                 (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
      except:
        pass
    # top_idx = [0, 1, 2, 3]
  return image

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.concatenate((x, y, z), axis=-1)
  return pt_3d

def unproject_2d_to_3d_torch(pt_2d, depth, P):
  # pts_2d: K x 2
  # depth: K x 1
  # P: 3 x 4
  # return: K x 3
  # Note: K can be zero for a single object but also nonzero for all objects in an image
  z = depth - P[2, 3]
  x = (pt_2d[...,0:1] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[...,1:2] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = torch.cat((x, y, z), axis=-1)
  return pt_3d


def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180

    :param alpha : Observation angle of object, ranging [-pi..pi], also called local orientation (see Mousavian)
    :param x : Object center x to the camera center (x-W/2), in pixels
    :param cx: principal point in x
    :param fx: focal length in x

    :return: rot_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    rot_y = np.where(rot_y >  math.pi, rot_y - 2 * np.pi, rot_y)
    rot_y = np.where(rot_y < -math.pi, rot_y + 2 * np.pi, rot_y)
    return rot_y

def alpha2rot_y_torch(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180

    :param alpha : Observation angle of object, ranging [-pi..pi], also called local orientation (see Mousavian)
    :param x : Object center x 
    :param cx: principal point in x
    :param fx: focal length in x

    :return: rot_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + torch.atan2(x - cx, fx) # rot_y = alpha + theta 
    rot_y = torch.where(rot_y >  math.pi, rot_y - 2 * np.pi, rot_y)
    rot_y = torch.where(rot_y < -math.pi, rot_y + 2 * np.pi, rot_y)
    return rot_y
    

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get alpha by rotation_y - theta + 180 where theta is angle of ray through object to camera CS
    
    :param rot_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    :param x : Object center x to the camera center (x-W/2), in pixels
    :param cx: principal point in x
    :param fx: focal length in x
    
    :return:  alpha : Observation angle of object, ranging [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx) # alpha = rot_y - theta
    # Put alpha in range [-pi,pi]
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot(center, alpha, dim, depth, calib):
  # single object
  locations = unproject_2d_to_3d(center, depth, calib)
  locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def ddd2locrot_torch(center, alpha, dim, depth, calib):
  # single or multiple objects
  locations = unproject_2d_to_3d_torch(center, depth, calib)
  locations[...,1] += dim[...,0] / 2
  rotation_y = alpha2rot_y_torch(alpha, center[...,0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
  box_3d = compute_box_3d(dim, location, rotation_y)
  box_2d = project_to_image(box_3d, calib)
  return box_2d

#-----------------------------------------------------
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def iou3d(corners1, corners2):
  ''' Compute 3D bounding box IoU.
  Input:
      corners1: numpy array (8,3), assume up direction is negative Y
      corners2: numpy array (8,3), assume up direction is negative Y
  Output:
      iou: 3D bounding box IoU
      iou_2d: bird's eye view 2D bounding box IoU
  '''
  # corner points are in counter clockwise order
  rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
  rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
  area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
  area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
  inter, inter_area = convex_hull_intersection(rect1, rect2)
  iou_2d = inter_area/(area1+area2-inter_area)
  ymax = min(corners1[0,1], corners2[0,1])
  ymin = max(corners1[4,1], corners2[4,1])
  inter_vol = inter_area * max(0.0, ymax-ymin)
  vol1 = box3d_vol(corners1)
  vol2 = box3d_vol(corners2)
  iou = inter_vol / (vol1 + vol2 - inter_vol)
  return iou, iou_2d


def iou3d_global(corners1, corners2):
  ''' Compute 3D bounding box IoU.
  Input:
      corners1: numpy array (8,3), assume up direction is negative Y
      corners2: numpy array (8,3), assume up direction is negative Y
  Output:
      iou: 3D bounding box IoU
      iou_2d: bird's eye view 2D bounding box IoU
  '''
  # corner points are in counter clockwise order
  rect1 = corners1[:,[0,3,7,4]].T
  rect2 = corners2[:,[0,3,7,4]].T
  
  rect1 = [(rect1[i,0], rect1[i,1]) for i in range(3,-1,-1)]
  rect2 = [(rect2[i,0], rect2[i,1]) for i in range(3,-1,-1)]
  
  area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
  area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
  inter, inter_area = convex_hull_intersection(rect1, rect2)

  iou_2d = inter_area/(area1+area2-inter_area)
  
  iou = 0
  # ymax = min(corners1[0,2], corners2[0,2])
  # ymin = max(corners1[1,2], corners2[1,2])
  # inter_vol = inter_area * max(0.0, ymax-ymin)
  # vol1 = box3d_vol(corners1)
  # vol2 = box3d_vol(corners2)
  # iou = inter_vol / (vol1 + vol2 - inter_vol)
  return iou, iou_2d

if __name__ == '__main__':
  calib = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  alpha = -0.20
  tl = np.array([712.40, 143.00], dtype=np.float32)
  br = np.array([810.73, 307.92], dtype=np.float32)
  ct = (tl + br) / 2
  rotation_y = 0.01
  print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
  print('rotation_y', rotation_y)
  
  
  # Test iou3d (camera coord)
  box_1 = compute_box_3d(dim=[1.599275 , 1.9106505, 4.5931444],
                         location=[ 7.1180778,  2.1364648, 41.784885],
                         rotation_y= -1.1312813047259618)
  box_2 = compute_box_3d(dim=[1.599275 , 1.9106505, 4.5931444],
                         location=[ 7.1180778,  2.1364648, 41.784885],
                         rotation_y= -1.1312813047259618)
  iou = iou3d(box_1, box_2)
  print("Results should be almost 1.0: ", iou)
  
  # # Test iou3d (global coord)  
  translation1 = [634.7540893554688, 1620.952880859375, 0.4360223412513733]
  size1 = [1.9073231220245361, 4.5971598625183105, 1.5940513610839844]
  rotation1 = [-0.6379619591303222, 0.6256341359192967, -0.320485847319929, 0.31444441216651253]
  
  translation2 = [634.7540893554688, 1620.952880859375, 0.4360223412513733]
  size2 = [1.9073231220245361, 4.5971598625183105, 1.5940513610839844]
  rotation2 = [-0.6379619591303222, 0.6256341359192967, -0.320485847319929, 0.31444441216651253]
  
  box_1 = Box(translation1, size1, Quaternion(rotation1))
  box_2 = Box(translation2, size2, Quaternion(rotation2))
  iou, iou_2d = iou3d_global(box_1.corners(), box_2.corners())
  print(iou, iou_2d)
  
