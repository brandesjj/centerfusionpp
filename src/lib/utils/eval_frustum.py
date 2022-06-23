import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle # Used to dump list 
from matplotlib.colors import Normalize
from typing import Tuple, List, Dict
import os
from random import randrange

import time

from utils.ddd_utils import compute_box_3d, vgt_to_vrad, calc_frustum_corners

import copy


def debug_lfa_frustum(obj, out, snap, ax, opt, phase):
    """
    Function to plot the frustum association using LFANet

    :param obj: Training obj
    :param out: Output of LFANet
    :param snap: Snap produced by LFANet
    :param ax: Matplotlib axis to plot on
    :param opt: Options
    :param phase: str 'val' or 'train'

    """

    if 'x' in opt.snap_channels:
        draw_points = True
        idx_x = opt.snap_channels.index('x')
    else:
        draw_points = False
    
    ax.grid()
    
    idx_z = opt.snap_channels.index('z')
    idx_vx = opt.snap_channels.index('vx_comp')
    idx_vz = opt.snap_channels.index('vz_comp')

    # Convert to numpy arrays
    snap = snap.cpu().detach().numpy().squeeze()
    out = out.cpu().detach().numpy()
    dim = obj['dim'].cpu().detach().numpy().squeeze()
    ann_pos = obj['location'].cpu().detach().numpy().squeeze()
    rot_y = obj['rotation_y'].cpu().detach().numpy().squeeze()
    if phase == 'train':
        v_gt = obj['velocity'].cpu().detach().numpy().squeeze()

    # Get radar points out of generated snap

    # Find the unique indices with the use of one feature:
    indices = np.unique(snap[0,:,:], return_index=True)[1]
    # Delete indices that correspond to zero values in snap
    indices = indices[np.where(snap.flatten()[indices]!=0)]

    # Get list of all radar points and their coordinates in the camera frame
    # radar_points[i]:

    radar_points = np.zeros((snap.shape[0], indices.shape[0]))
    for i in range(snap.shape[0]):
        radar_points[i,:] = snap[i,:,:].flatten()[indices]

    if opt.normalize_depth_input_lfa:
        radar_points[idx_z,:] *= opt.max_pc_depth

    # Scatter radar points

    # Plot radar points and  velocities
    label_once = True
    for idx in range(radar_points.shape[1]):
        if draw_points:
            ax.scatter(radar_points[idx_x,idx], radar_points[idx_z,idx])
            ax.arrow(radar_points[idx_x,idx], radar_points[idx_z,idx], radar_points[idx_vx,idx], radar_points[idx_vz,idx], color='b', alpha=0.4)
        else:
            # Draw horizontal line since x not in snap
            label_dep = 'Points depth' if label_once else ''
            ax.axhline(radar_points[idx_z,idx], label=label_dep)
            # Draw velocity from center point in x and depth in z
            label_vel = 'Points Vel' if label_once else ''
            ax.arrow(ann_pos[[0,2]][0], radar_points[idx_z,idx], radar_points[idx_vx,idx], radar_points[idx_vz,idx], color='b', alpha=0.4, label=label_vel)
            label_once = False
    
    # Draw horizontal line representing output
    ax.axhline(out[0], label='Output depth', color='#0bda51')

    # Plot bounding box
    # Compute 3D bounding box corners
    corners_3d = compute_box_3d(dim, ann_pos, rot_y)
    # Compute 2D bounding box corners in BEV
    corners_2d = corners_3d[0:4][:,[0,2]]
    # Draw bounding box in BEV
    label = 'GT 2D BBOX'
    ax.plot(corners_2d.T[0,[0,1]], corners_2d.T[1,[0,1]], color='c', label=label)
    ax.plot(corners_2d.T[0,[1,2]], corners_2d.T[1,[1,2]], color='c')
    ax.plot(corners_2d.T[0,[2,3]], corners_2d.T[1,[2,3]], color='c')
    ax.plot(corners_2d.T[0,[3,0]], corners_2d.T[1,[3,0]], color='c')

    # Draw CenterPoint of annotation
    ax.scatter(ann_pos[[0,2]][0], ann_pos[[0,2]][1], marker='+', color = 'c', s=50, label=label)

    if phase == 'train':
        # Draw ground truth velocity
        ax.arrow(ann_pos[0], ann_pos[2], v_gt[0], v_gt[2], color='k', length_includes_head=True, label='GT Velocity', head_width=0.05)
        
        vel_gt_rad = vgt_to_vrad(v_gt[[0,2]], ann_pos[[0,2]])
    
        if np.linalg.norm(vel_gt_rad) > np.linalg.norm(out[[1,2]]):
            # Ground truth velocity bigger -> plot first
            # Draw projected ground truth velocity
            ax.arrow(ann_pos[0], ann_pos[2], vel_gt_rad[0], vel_gt_rad[1], color='b', length_includes_head=True, label='GT Rad. Vel', head_width=0.05)

            # Draw detected velocity
            ax.arrow(ann_pos[0], out[0], out[1], out[2], color='r', length_includes_head=True, label='LFANet Velocity', head_width=0.05)
        else:
            # Draw detected velocity
            ax.arrow(ann_pos[0], out[0], out[1], out[2], color='r', length_includes_head=True, label='LFANet Velocity', head_width=0.05)

            # Draw projected ground truth velocity
            ax.arrow(ann_pos[0], ann_pos[2], vel_gt_rad[0], vel_gt_rad[1], color='b', length_includes_head=True, label='GT Rad. Vel', head_width=0.05)
    else:
        # Draw only output velocity
        ax.arrow(ann_pos[0], out[0], out[1], out[2], color='r', length_includes_head=True, label='LFANet Velocity', head_width=0.05)

    ax.axis('equal')
    ax.legend()

class EvalFrustum():

    def __init__(self, opt=None):
        if opt is not None:
            self.opt = opt
            self.num_correct_ass = 0
            self.num_wrong_ass = 0
            self.err_vr = np.empty((1,))
            self.err_dep = np.empty((1,))
            self.err_vr_norm = np.empty((1,))
            self.err_dep_norm = np.empty((1,))
            self.dist_thresh_nabati = 0 # Distance threshold with error (used to visualize)
            self.dist_thresh_depth = 0
            self.dist_thresh_dist = 0
            self.nr_saved_plots = 0

        self.last_dist_tresh = 0
        self.last_vr_tresh = 0

        if opt.eval_frustum == 5:
            self.frustum_dicts = list()
            self.frustum_dict_keys = ['num_points', 'category', 'category_id', 'bbox_size', 'frustum_size', 'depth_to_center']
            assert not self.opt.use_dist_for_frustum, 'Snapshot evaluation not implemented for distance yet. Use depth.'

        self.categories = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
                        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

    def add(self, res):
        # If results are empty, skip
        if res == None:
            return
        # Add results to the evaluation
        if res['idx_best'] == res['idx_sel']:
            self.num_correct_ass += 1
        else:
            self.num_wrong_ass += 1
            self.err_vr = np.append(self.err_vr, res['err_vr_rel'][res['idx_sel']])
            self.err_dep = np.append(self.err_dep, res['err_dep_rel'][res['idx_sel']])
            self.err_vr_norm = np.append(self.err_vr_norm, res['err_vr_norm'][res['idx_sel']])
            self.err_dep_norm = np.append(self.err_dep_norm, res['err_dep_norm'][res['idx_sel']])

        # opt.frustum_eval['wrong_res'].append(res)

    def print(self):
        print(f'\nFrustum association correct in {self.num_correct_ass} of {self.num_correct_ass+self.num_wrong_ass} cases\n')

    def plot_histograms(self,
                        save_plots = False,
                        ):
        """
        Plot histograms of errors when wrong point is selected
        """

        # fig, axes = plt.subplots(2,2, tight_layout=True)
        
        # axes[0,0].hist(self.err_vr, 100)
        # axes[0,0].title.set_text('Error in relative velocity to best point')
        # axes[1,0].hist(self.err_dep, 100)
        # axes[1,0].title.set_text('Error in position to best point')
        # axes[0,1].hist(self.err_vr, 100)
        # axes[0,1].title.set_text('Norm error in relative vel to best')
        # axes[1,1].hist(self.err_dep, 100)
        # axes[1,1].title.set_text('Norm error in pos to best')

        # axes[0,0].grid()
        # axes[0,1].grid()
        # axes[1,0].grid()
        # axes[1,1].grid()

        # if save_plots:
        #     os.makedirs(self.opt.log_dir + '/plots/', exist_ok=True)
        #     plt.savefig(self.opt.log_dir + '/plots/histogram_' + self.opt.train_split + time.strftime('%Y-%m-%d-%H-%M') + '.pdf')
        # else:
        #     plt.show()


        errors = [self.err_vr, self.err_dep, self.err_vr_norm, self.err_dep_norm]
        titles = ['Error in relative velocity to best point', 'Error in position to best point',
                  'Norm error in relative vel to best', 'Norm error in pos to best']
        for i in range(4):
            fig, ax = plt.subplots(1,1,figsize=(5,3))            
            ax.hist(errors[i], 100)
            ax.title.set_text(titles[i])
            plt.yscale('log')
            ax.grid()
            if save_plots:
                os.makedirs(self.opt.log_dir + '/plots/', exist_ok=True)
                plt.savefig(self.opt.log_dir + '/plots/histogram_' + self.opt.train_split + time.strftime('%Y-%m-%d-%H-%M') + '_' + str(i) + '.pdf')
            else:
                plt.show()
  

    def plot_vgt_vs_vrad(self,
                         ax: plt.axis, 
                         vel_gt_rads: List,
                         vel_gt: np.float32,
                         vel_gt_rad: np.float32,
                         vel_rad: List,
                         radar_pos: List,
                         ann_pos: np.array,
                         category: int,
                         idx_selected: int,
                         idx_best: int,
                         rcs,
                         dim,
                         rot_y,
                         frustum_thresh=None):
        """
        Plot radial velocity of a given point and compare it with the ground truth of the annotation 
        and the calculated ground truth of the radar point

        :param ax: Matplotlib axis to plot on
        :param vel_gt_rads: List of estimated ground truth radial velocities of the radar points
        :param vel_gt: Ground truth velocity of the annotation
        :param vel_gt_rad: Ground truth radial velocity of the annotation
        :param vel_rad: List of radial velocities of the radar points
        :param radar_pos: List of positions of the radar points in the camera frame
        :param ann_pos: 3D Position of the annotated object in the camera frame 
        :param category: Category-ID of the object
        :param idx_selected: Index within given lists for value selected by frustum_association
        :param idx_best: "Best" index to choose according to some metric
        :param rcs: List of RCS values of the current radar points
        :param dim: array of ground truth 3D bounding box dimensions
        :param rot_y: scalar of ground truth rotation angle around y-axis of camera CS
        :param frustum_thresh: Distance/Depth threshold of frustum which should be plotted
        """
        ax.grid()
        
        frustum_xlim = np.array([1e4,-1e4]) # min and max
        frustum_zlim = np.array([1e4,-1e4]) # min and max
        # Compute 3D bounding box corners
        corners_3d = compute_box_3d(dim, ann_pos, rot_y)
        # Compute 2D bounding box corners in BEV
        corners_2d = corners_3d[0:4][:,[0,2]]
        # Draw bounding box in BEV
        label = 'GT 2D-BB'
        ax.plot(corners_2d.T[0,[0,1]], corners_2d.T[1,[0,1]], color='c', label=label)
        ax.plot(corners_2d.T[0,[1,2]], corners_2d.T[1,[1,2]], color='c')
        ax.plot(corners_2d.T[0,[2,3]], corners_2d.T[1,[2,3]], color='c')
        ax.plot(corners_2d.T[0,[3,0]], corners_2d.T[1,[3,0]], color='c')

        # Draw CenterPoint of annotation
        label = f'Center of {category}'
        ax.scatter(ann_pos[[0,2]][0], ann_pos[[0,2]][1], marker='+', color = 'c', s=50, label=label)

        # Plot Frustum
        if self.opt.use_dist_for_frustum:

            opt_temp = copy.deepcopy(self.opt)
            opt_temp.use_dist_for_frustum = True
            if frustum_thresh == None:
                thresh = self.dist_thresh_dist
            else:
                thresh = frustum_thresh
            frustum_corners = calc_frustum_corners(thresh, np.linalg.norm(ann_pos[[0,2]]), corners_2d, opt_temp) 
            # frustum_nabati_corners = calc_frustum_corners(self.dist_thresh_nabati, ann_pos[2], corners_2d, self.opt) # Uses depth as ann dist
        
            frustum_xlim = np.array([np.amin(np.concatenate((np.array([frustum_xlim[0]]), frustum_corners[:,0]))), np.amax(np.concatenate((np.array([frustum_xlim[1]]), frustum_corners[:,0])))])
            frustum_zlim = np.array([np.amin(np.concatenate((np.array([frustum_zlim[0]]), frustum_corners[:,1]))), np.amax(np.concatenate((np.array([frustum_zlim[1]]), frustum_corners[:,1])))])

            # Plot the frustum with dist
            for id in np.arange(frustum_corners.shape[0]):
                if id in [0,2]:
                    # Draw arc
                    theta1 = np.rad2deg(np.arctan(frustum_corners[id,0] / frustum_corners[id,1]))
                    theta2 = np.rad2deg(np.arctan(frustum_corners[id+1,0] / frustum_corners[id+1,1]))
                    thetas = np.array([theta1%360, theta2%360])
                    theta1 = np.min(thetas)
                    theta2 = np.max(thetas)
                    xlim = np.array(ax.get_xlim())
                    ylim = np.array(ax.get_ylim())

                    arc1 = matplotlib.patches.Arc((0,0), 2*np.linalg.norm(frustum_corners[0,:]), 2*np.linalg.norm(frustum_corners[0,:]), 90, -theta2, -theta1, color='green')
                    arc2 = matplotlib.patches.Arc((0,0), 2*np.linalg.norm(frustum_corners[2,:]), 2*np.linalg.norm(frustum_corners[2,:]), 90, -theta2, -theta1, color='green', label='Distance Frustum')
                    ax.add_patch(arc1)
                    ax.add_patch(arc2)
                    ax.set_xlim(xlim) # Reset xlim
                    ax.set_ylim(ylim) # Reset ylim
                    # ax.draw(arc2)

                else:
                    # Draw lin
                    if id == 1:
                        ax.plot([frustum_corners[id,0], frustum_corners[id+1,0]], [frustum_corners[id,1], frustum_corners[id+1,1]], color='g')
                    else:
                        ax.plot([frustum_corners[id,0], frustum_corners[0,0]], [frustum_corners[id,1], frustum_corners[0,1]], color='g')
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
        
        else:

            opt_temp = copy.deepcopy(self.opt)
            opt_temp.use_dist_for_frustum = False
            if frustum_thresh == None:
                thresh = self.dist_thresh_depth
            else:
                thresh = frustum_thresh
            frustum_depth_corners = calc_frustum_corners(thresh, ann_pos[2], corners_2d, opt_temp) # Uses depth as ann dist

            frustum_xlim = np.array([np.amin(np.concatenate((np.array([frustum_xlim[0]]), frustum_depth_corners[:,0]))), np.amax(np.concatenate((np.array([frustum_xlim[1]]), frustum_depth_corners[:,0])))])
            frustum_zlim = np.array([np.amin(np.concatenate((np.array([frustum_zlim[0]]), frustum_depth_corners[:,1]))), np.amax(np.concatenate((np.array([frustum_zlim[1]]), frustum_depth_corners[:,1])))])


            # Plot the frustum like Nabati (depth), without ) error
            label_bool_frustum = True
            for id in np.arange(frustum_depth_corners.shape[0]):
                label = 'Depth Frustum' if label_bool_frustum else ''
                if id+1 < frustum_depth_corners.shape[0]:
                    ax.plot([frustum_depth_corners[id,0], frustum_depth_corners[id+1,0]], [frustum_depth_corners[id,1], frustum_depth_corners[id+1,1]], label=label, color='orange', alpha=0.5)
                else:
                    ax.plot([frustum_depth_corners[id,0], frustum_depth_corners[0,0]], [frustum_depth_corners[id,1], frustum_depth_corners[0,1]], label=label, color='orange', alpha=0.5)
                label_bool_frustum = False
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                    
        # Draw gt velocity from centerpoint
        label = r'$v$'
        ax.arrow(ann_pos[0], ann_pos[2], vel_gt[0], vel_gt[1], color='k', length_includes_head=True, label=label, head_width=0.1)
        
        # Draw gt radial velocity from ct point
        label = r'$v_{rad}$'
        ax.arrow(ann_pos[0], ann_pos[2], vel_gt_rad[0], vel_gt_rad[1], color='b', length_includes_head=True, label=label, head_width=0.1)

        # Compute rcs range
        rcs_range = [np.amin(rcs), np.amax(rcs)]

        # Iterate over all given radar points to plot them
        label_bool = True
        for idx in np.arange(len(vel_rad)):

            # label_bool = True if idx==0 else False # Label only the first point plotted

            # Draw tangents of radial velocity

            # R = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]]) # rotates by 90 deg
            # dest = vel_gt_rads @ R # rotate radial velocity by 90 deg
            # if dest[0]+vel_gt_rads[0] != 0 or dest[1]+vel_gt_rads[1] != 0:
            #   ax.axline((radar_pos[0]+vel_gt_rads[0], radar_pos[1]+vel_gt_rads[1]), (radar_pos[0]+vel_gt_rads[0]+dest[0], radar_pos[1]+vel_gt_rads[1]+dest[1]), color='b', alpha=0.4)

            # Draw extended ground truth velocity vector
            
            # if radar_pos[0]+vel_gt_rads[0] != 0 or radar_pos[1]+vel_gt_rads[1] != 0:
            #   ax.axline((0, 0), (radar_pos[0]+vel_gt_rads[0], radar_pos[1]+vel_gt_rads[1]), color='k', alpha=0.2)

            # Draw gt velocity

            # label = 'GT_vel' if label_bool else ''
            # if radar_pos[idx][0] != v_gt[0]:
            #     ax.arrow(radar_pos[idx][0], radar_pos[idx][1], v_gt[0], v_gt[1], color='k', length_includes_head=True, label=label, head_width=0.002)
            
            # Draw extended radial velocity vector

            # if radar_pos[idx][0] != radar_pos[idx][0]+vel_rad[idx][0]:
            #     ax.axline((radar_pos[idx][0], radar_pos[idx][1]), (radar_pos[idx][0]+vel_rad[idx][0], radar_pos[idx][1]+vel_rad[idx][1]), color='r', alpha=0.2)

            # Make sure both velocities are visible since both are plotted in the same direction with the following if:else
            # if np.linalg.norm(vel_gt_rads) > np.linalg.norm(vel_rad):
            #     # Draw GT radial velocity
            #     if vel_gt_rads[idx][0] != 0 or vel_gt_rads[idx][1] != 0:
            #         label = 'Radial GT vel' if label_bool else ''
            #         ax.arrow(radar_pos[idx][0], radar_pos[idx][1], vel_gt_rads[idx][0], vel_gt_rads[idx][1], color='b', length_includes_head=True, label=label, head_width=0.1)
            #     # Draw actual radial velocity from radar point
            #     if vel_rad[idx][0] != 0 or vel_rad[idx][1] != 0:
            #         label = 'Radial vel' if label_bool else ''
            #         ax.arrow(radar_pos[idx][0], radar_pos[idx][1], vel_rad[idx][0], vel_rad[idx][1], color='r', length_includes_head=True, label=label, head_width=0.1)
            # else: 
            #     # Draw actual radial velocity from radar point
            #     if vel_rad[idx][0] != 0 or vel_rad[idx][1] != 0:
            #         label = 'Radial vel' if label_bool else ''
            #         ax.arrow(radar_pos[idx][0], radar_pos[idx][1], vel_rad[idx][0], vel_rad[idx][1], color='r', length_includes_head=True, label=label, head_width=0.1)
            #     # Draw GT radial velocity
            #     if vel_gt_rads[idx][0] != 0 or vel_gt_rads[idx][1] != 0:
            #         label = 'Radial GT vel' if label_bool else ''
            #         ax.arrow(radar_pos[idx][0], radar_pos[idx][1], vel_gt_rads[idx][0], vel_gt_rad[idx][1], color='b', length_includes_head=True, label=label, head_width=0.1)
            
            # # Draw actual radial velocity from radar point
            # label = r'$v^{(r)}$' if label_bool else ''
            # ax.arrow(radar_pos[idx][0], radar_pos[idx][1], vel_rad[idx][0], vel_rad[idx][1], color='r', length_includes_head=True, label=label, head_width=0.1)
            
            # Select radar point label
            label = 'Best Radar point' if idx == idx_best else ''
            label = 'Selected Radar point' if idx == idx_selected else label
            label = 'Selected & Best point' if (idx == idx_selected and idx == idx_best) else label
            # Draw selected radar point as square, best as filled cross or as a star in case the selected point is also the best
            marker = 's' if idx == idx_selected else 'o'
            marker = 'P' if idx == idx_best else marker
            marker = '*' if (idx == idx_selected and idx == idx_best) else marker
            # Plot radar point using the RCS value
            rcs_range = [np.amin(rcs), np.amax(rcs)]
            # ax.scatter(radar_pos[idx][0], radar_pos[idx][1], c = rcs[idx], cmap = 'autumn', edgecolor='k',vmin=rcs_range[0], vmax=rcs_range[1], marker = marker, label=label, s=60)
            
            # Plot radar point using the RCS value
            if idx != idx_best and idx != idx_selected:
                # # Draw actual radial velocity from radar point
                label = r'$v^{(r)}$' if label_bool else ''
                ax.arrow(radar_pos[idx][0], radar_pos[idx][1], vel_rad[idx][0], vel_rad[idx][1], color='r', length_includes_head=True, label=label, head_width=0.1)
                # Draw radar point
                marker = 'o'
                ax.scatter(radar_pos[idx][0], radar_pos[idx][1], c = rcs[idx], cmap = 'autumn', edgecolor='k',vmin=rcs_range[0], vmax=rcs_range[1], marker = marker, s=60)
                # ax.scatter(radar_pos[idx][0], radar_pos[idx][1], color = 'r', edgecolor='k', marker = marker, s=60)
                label_bool = False
        
        # Plot special points last and their velocities
        label_selected = 'Selected Radar point'
        label_best = 'Best Radar point'
        label_optimal = 'Selected & Best point'
        marker_selected = 's'
        marker_best = 'P'   
        marker_optimal = '*'
        label = r'$v^{(r)}$' if label_bool else ''
        if idx_selected == idx_best:
            ax.scatter(radar_pos[idx_selected][0], radar_pos[idx_selected][1], c = rcs[idx_selected], cmap = 'autumn', edgecolor='k',vmin=rcs_range[0], vmax=rcs_range[1], marker = marker_optimal, label=label_optimal, s=60)

            ax.arrow(radar_pos[idx_selected][0], radar_pos[idx_selected][1], vel_rad[idx_selected][0], vel_rad[idx_selected][1], color='r', length_includes_head=True, label=label, head_width=0.1)
                
        else:
            ax.arrow(radar_pos[idx_selected][0], radar_pos[idx_selected][1], vel_rad[idx_selected][0], vel_rad[idx_selected][1], color='r', length_includes_head=True, label=label, head_width=0.1)
            ax.arrow(radar_pos[idx_best][0], radar_pos[idx_best][1], vel_rad[idx_best][0], vel_rad[idx_best][1], color='r', length_includes_head=True, head_width=0.1)
            
            ax.scatter(radar_pos[idx_selected][0], radar_pos[idx_selected][1], c = rcs[idx_selected], cmap = 'autumn', edgecolor='k',vmin=rcs_range[0], vmax=rcs_range[1], marker = marker_selected, label=label_selected, s=60)
            ax.scatter(radar_pos[idx_best][0], radar_pos[idx_best][1], c = rcs[idx_best], cmap = 'autumn', edgecolor='k',vmin=rcs_range[0], vmax=rcs_range[1], marker = marker_best, label=label_best, s=60)

        
        # Draw angle through frustum center
        # theta = -(np.arctan2(frustum_corners[1,0],frustum_corners[1,1]) + np.arctan2(frustum_corners[0,0],frustum_corners[0,1]))/2+np.pi/2
        # ax.axline((0,0), (1, np.tan(theta)*1), color='r')

        ax.legend()
        ax.axis('equal')
        

        ax.set_xlim((frustum_xlim[0]-1, frustum_xlim[1]+1))
        ax.set_ylim((frustum_zlim[0]-1, frustum_zlim[1]+1))

    def check_savefig(self, res):
        """
        When the selected value is far enough off from the best value, return True to save the fig
        """

        dist_thresh = 1 # Distane threshold [m]
        vr_thres = 1 # Radial velocity threshold [m/s]

        check_dist = res['err_dep_rel'][res['idx_sel']] > max(self.last_dist_tresh, dist_thresh)
        check_vr = res['err_vr_rel'][res['idx_sel']] > max(self.last_vr_tresh, vr_thres)
        
        if check_dist:
            self.last_dist_tresh = res['err_dep_rel'][res['idx_sel']]
        if check_vr:
            self.last_vr_tresh = res['err_vr_rel'][res['idx_sel']]


        return check_dist or check_vr

    def position_cost(self, pos_radar, pos_gt, dim, rot_y):
        """
        Compute the positional cost of chosen radar point relative to the boundaries of
        the 3D bounding box. A "good" radar point has a low positional cost since
        automotive radar hit objects on their boundaries. If a radar point is far away 
        from the boundary it's likely a false positive detection.
        
        :param pos_radar: array of 2D birds eye view position of radar point
        :param pos_gt: array of 2D birds eye view position of the ground truth center point
        :param dim: array of ground truth 3D bounding box
        :param rot_y: scalar of ground truth rotation angle around y-axis of camera CS

        :return: pos_cost: scalar with positional cost of radar point
        """

        pos_cost = 0
        h = 1e4 # just a very big number
        inside = 1

        # Compute 3D bounding box corners
        corners_3d = compute_box_3d(dim, pos_gt, rot_y)

        # Compute 2D bounding box corners in BEV
        corners_2d = corners_3d[0:4][:,[0,2]]

        # Compute cost of half-spaces
        for edge in range(len(corners_2d)):
            x1 = corners_2d[edge][0]
            z1 = corners_2d[edge][1]
            x2 = corners_2d[(edge+1)%len(corners_2d)][0] # loop over if edge is at last index take first one as next vertex
            z2 = corners_2d[(edge+1)%len(corners_2d)][1] # loop over if edge is at last index take first one as next vertex
            deltax = x2 - x1
            deltay = z1 - z2
            norm = np.linalg.norm((deltax,deltay)) + 1e-9  # to normalize cost value (has to be nonzero for numerical stability)
            # Represent edge by a line
            a0 = - ( z1 - z2 )/norm  # compute a0 for current edge
            a1 = - ( x2 - x1 )/norm  # compute a1 for current edge
            b = - ( z1*x2 - z2*x1 )/norm
            # Evaluate half-space of each edge at radar point
            half_space_eq = b - a0*pos_radar[0] - a1*pos_radar[1]
            
            # Store minimal abs cost over half-spaces 
            h = min(h, abs(half_space_eq))
            
            # If point is outside w.r.t. one half space set inside to zero
            inside *= max(0.0, half_space_eq)
        # print("Inside: ", inside)
        # If point is inside, weight the cost less
        weight = 0.5 if inside > 0 else 1
        pos_cost = weight * h
        # print("Minimal edge cost h: ", h)
        # print("Final pos_cost: ", pos_cost)
        return pos_cost

    def analyze_frustum_association(self,
                                    pc_pos_x_match,
                                    pc_dep_match, 
                                    pc_vx_match,
                                    pc_vz_match,
                                    pc_rcs_match, # Currently not in use
                                    pc_dts_match, # Currently not in use
                                    idx_selection,
                                    ann,
                                    frustum_dist_thresh=None,
                                    ):
        """
        Analyze whether the radar point selected by the frustum association method was a valid / good choice.
        
        :param pc_pos_x_match: array of position in x (Camera CS) values of radar points within the frustum
        :param pc_dep_match: array of depth values of radar points within the frustum
        :param pc_vx_match: array of velocity x values of radar points within the frustum
        :param pc_vz_match: array of velocity z values of radar points within the frustum
        :param pc_rcs_match: array of RCS values of radar points within the frustum
        :param pc_dts_match: array of delta timestamp values of radar points within the frustum
        :param idx_selection: Index of the point selected by the frustum association within the above arrays
        :param ann: Annotation of the current object

        :return: res: dict with keys:
                        - 'idx_best': int, index of best radar point according to our metric for the error arrays
                        - 'idx_sel': int, index of actually selected point for the error arrays
                        - 'err_vr': np.array, absolut error in radial velocity
                        - 'err_dep': np.array, absolut error from center point
                        - 'err_vr_rel': np.array, relative error in radial velocity to best point in radial vel
                        - 'err_dep_rel': np.array, relative error in position to best point in position 
                    Return None in case the association is not analyze (e.g. when there are only 2 points)

        """


        categories = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
                        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

        res = dict.fromkeys(['idx_best', 'idx_sel', 'err_vr', 'err_dep', 'err_vr_rel', 'err_dep_rel'])
        res['err_vr'] = []
        res['err_dep'] = []

        # Since the inputs are heatmaps of pillars there are a lot of duplicate values. 
        # Find the unique indices with the use of one feature:
        _, indices = np.unique(pc_dep_match, return_index=True) 
        
        # The index of the selection might have changed due to deletion of duplicates. 
        # The following handles that problem
        for index in indices:
            if pc_dep_match[index] == pc_dep_match[idx_selection]:
                idx_selection = index
                continue

        # Only analyze the frustum if there is more than 1 point
        # Add nothing to evaluation method.
        if len(indices) < 2:
            return

        vel_gt_rads = [] # Gt radial vel in all radar directions
        vel_rad = []
        pos_radar = []
        rcs = []

        # Ground truth 3D position of the objects centerpoint in camera CS
        pos_gt = np.array(ann['location'])

        # Ground truth velocity of annotated object in camera frame.
        # Only get x,z component of velocity vector. Since we are not interested in learning
        # from the y component.
        vel_gt = np.array(ann['velocity_cam'])[[0,2]] 
        vel_gt_rad = vgt_to_vrad(vel_gt, pos_gt[[0,2]])

        # Ground truth dimensions of 3D bounding box 
        dim = np.array(ann['dim'])

        # Ground truth yaw angle
        rot_y = np.array(ann['rotation_y'])
        # Iterate over all radar points in the frustum
        for point_idx in indices:
            # Position of radar in camera frame
            pos_radar.append(np.array([pc_pos_x_match[point_idx], pc_dep_match[point_idx]]))

            # Radial velocity vector in camera CS
            vel_rad.append(np.array([pc_vx_match[point_idx], pc_vz_match[point_idx]]))
            # Ground truth radial velocity in direction of radar points
            vel_gt_rads.append(vgt_to_vrad(vel_gt, vel_rad[-1]))
            # RCS value (only relevant for plotting)
            rcs.append(pc_rcs_match[point_idx])

            # Calculate radial velocity error
            res['err_vr'].append(np.linalg.norm(vel_gt_rad - vel_rad[-1]))
            # Calculate translation error (error between radar point and objects centerpoint)
            # Positional error as distance to boundary of 3D bounding box
            # res['err_dep'].append(self.position_cost(pos_radar[-1], pos_gt, dim, rot_y))
            # Positional error to center point
            res['err_dep'].append(np.linalg.norm(pos_gt[[2]] - pos_radar[-1][1]))
            # Check whether current point is selected point   
            if point_idx == idx_selection:
                res['idx_sel'] = len(res['err_vr'])-1

        # Calculate relative errors to minimum error
        res['err_vr_rel'] = res['err_vr']-np.amin(res['err_vr'])
        res['err_dep_rel'] = res['err_dep']-np.amin(res['err_dep'])

        # Normalize errors
        res['err_vr_norm'] = res['err_vr_rel']/np.amax(res['err_vr_rel'])
        res['err_dep_norm'] = res['err_dep_rel']/np.amax(res['err_dep_rel'])

        # Calculate combined error of normalized errors
        # rad_norm is multiplied by 1.001 to put slightly more emphasis on it in case there are only 2 points
        res['err_norm'] = 1.001*res['err_vr_norm']+res['err_dep_norm']

        # Find best index according to our metric
        res['idx_best'] = np.argmin(res['err_norm'])

        # If required, plot the findings
        if self.opt.eval_frustum > 2:
            plt.ioff()
            if self.opt.use_lfa and 'cat' in self.opt.snap_channels:
                cat = categories[ann['cat']] # index (starts at 0)
            else:
                cat = categories[ann['category_id']-1] # id (starts at 1)
            if cat in ['car', 'truck', 'bus']: # Car
                fig, ax = plt.subplots(1,1,figsize=(10,8))
                self.plot_vgt_vs_vrad(ax, vel_gt_rads, vel_gt, vel_gt_rad, vel_rad, pos_radar, pos_gt, cat,
                                res['idx_sel'], res['idx_best'], rcs, dim, rot_y, frustum_thresh=frustum_dist_thresh)

                plt.colorbar(plt.cm.ScalarMappable(Normalize(vmin=np.amin(rcs),vmax=np.amax(rcs)), cmap='autumn'))
                plt.tight_layout()
                plt.xlabel('$X$')
                plt.ylabel('$Z$')

                ax.grid(b=True)
                if self.opt.eval_frustum < 5:
                    plt.show()
                    plt.figure().clf()
                    plt.figure().clear()
                elif self.opt.eval_frustum == 4:
                    # Check whether it makes sense to save current figure
                    if self.check_savefig(res) and res['idx_best'] != res['idx_sel']:
                        os.makedirs(self.opt.log_dir + '/plots/jpg', exist_ok=True)
                        os.makedirs(self.opt.log_dir + '/plots/svg', exist_ok=True)

                        try:
                            # Sometimes there is a problem with the line drawing, so this is supposed to catch the error
                            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S_')
                            rand_nr = str(randrange(100))
                            plot_index = str(self.nr_saved_plots)
                            plt.savefig(self.opt.log_dir + '/plots/jpg/frustum_' + self.opt.train_split + plot_index + '.jpg', dpi=500)
                            plt.savefig(self.opt.log_dir + '/plots/svg/frustum_' + self.opt.train_split + plot_index + '.svg')
                            self.nr_saved_plots += 1
                        except:
                            # Failed to save the plot, just ignore this.
                            pass
                plt.close("all")

        self.add(res)

    def eval_snapshot(self, snap, frustum_bounds, nr_frustum_points, ann):
        """
        Evaluate the snapshot. 
        The data is stored in a list of dicts and dumped using the pickle library
        after a full epoch.
        The following is stored:
        - bbox_size: Size of the bounding box in z direction (camera's frame)
        - frustum_size: Size of frustum in z direction (camera's frame)
        - num_points: Number of radar points within the Frustum
        - category_id: ID of the category
        - category: category of the object
        - depth_to_center: Difference in depth between center of annotation and each radar point

        """
        

        # Create empty dictonary 
        frustum_dict = dict.fromkeys(self.frustum_dict_keys)

        # Fill dict with category information
        frustum_dict['category_id'] = ann['category_id'] # -1 if looking up actual category
        frustum_dict['category'] = self.categories[ann['category_id']-1]
        # Store size of annotation
        # Compute 3D bounding box corners in BEV (planar)
        bbox_corners = compute_box_3d(ann['dim'], ann['location'], \
                                            ann['rotation_y'])
        bbox_corners = bbox_corners[0:4][:,[0,2]]        
        frustum_dict['bbox_size'] = np.amax(bbox_corners[:,1])-np.amin(bbox_corners[:,1])

        if nr_frustum_points == 0:
            # Empty snap
            frustum_dict['num_points'] = 0
            # Append frustum dict to list of dicts
            self.frustum_dicts.append(frustum_dict)
            return # Exit evaluation.

        if 'z' in self.opt.snap_channels:
            # Get number of unique radar points in snapshot
            point_depths, point_indices = np.unique(snap[self.opt.lfa_index_z,:,:], return_index=True) 
            # De-normalize depths
            if self.opt.normalize_depth:
                point_depths *= self.opt.max_pc_depth
        
        # Remove 0 from points
        non_zero = np.where(point_depths > 0)
        point_depths = point_depths[non_zero]

        frustum_dict['num_points'] = np.shape(point_depths)[0]
        # Calculate and store frustum_size
        frustum_dict['frustum_size'] = float(np.diff(frustum_bounds))
        # Calculate difference in depth between depth in snap center of bounding box
        frustum_dict['depth_to_center'] = (point_depths - ann['location'][2]).tolist()

        # Append frustum dict to list of dicts
        self.frustum_dicts.append(frustum_dict)

    def dump_snapshot_eval(self):
        """
        Dump the snapshot dict to log_dir
        """
        file_name = self.opt.log_dir+'/snapshot_eval.pkl'
        file = open(file_name, 'wb')
        pickle.dump(self.frustum_dicts, file)
        file.close()