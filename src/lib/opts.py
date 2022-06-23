from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from numpy import DataSource
import torch
import time
class opts(object):
  def __init__(self):
    
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='',
                             help='ctdet | ddd | multi_pose '
                             '| tracking or combined with ,')
    self.parser.add_argument('--dataset', default='nuscenes',
                             help='see lib/dataset/dataset_factory for ' + 
                            'available datasets')
    self.parser.add_argument('--test_dataset', default='',
                             help='coco | kitti | coco_hp | pascal')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--eval', action='store_true',
                             help='only evaluate the val split and quit')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk'
                            )
    self.parser.add_argument('--eval_frustum', type=int, default=0,
                             help='Level of Evaluation of Frustum association.'
                                  '0: do not evaluate the frustum association'
                                  '1: print number of correct/wrong associations'
                                  '2: plot histograms'
                                  '3: show the matplotlib plots of all frustums with more than 1 point'
                                  '4: save plots of big deviations and histograms. Do not plot them (does not include 1 to 3)'
                                  '5: Save statistics on frustums in a dictonary. Dumped using pickle.'
                            )
    self.parser.add_argument('--no_pause', action='store_true',
                             help='do not pause after debugging visualizations')
    self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    #######                  
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 

    # system
    self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet
    self.parser.add_argument('--not_set_cuda_env', action='store_true',
                             help='used when training in clusters.')

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    self.parser.add_argument('--run_dataset_eval', action='store_true',
                             help='use dataset specific evaluation function in eval')
    self.parser.add_argument('--save_imgs', default='',
                             help='list of images to save in debug. empty to save all')
    self.parser.add_argument('--save_img_suffix', default='', help='')
    self.parser.add_argument('--skip_first', type=int, default=-1,
                             help='skip first n images in demo mode')
    self.parser.add_argument('--save_video', action='store_true')
    self.parser.add_argument('--save_framerate', type=int, default=30)
    self.parser.add_argument('--resize_video', action='store_true')
    self.parser.add_argument('--video_h', type=int, default=512, help='')
    self.parser.add_argument('--video_w', type=int, default=512, help='')
    self.parser.add_argument('--transpose_video', action='store_true')
    self.parser.add_argument('--show_track_color', action='store_true')
    self.parser.add_argument('--not_show_bbox', action='store_true')
    self.parser.add_argument('--not_show_number', action='store_true')
    self.parser.add_argument('--qualitative', action='store_true')
    self.parser.add_argument('--tango_color', action='store_true')

    # model
    self.parser.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested'
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
    self.parser.add_argument('--dla_node', default='dcn',
                            help='Sets the node type in the DLA backbone. Options are'
                                 'dcn [DeformConv]| gcn [GlobalConv] | conv [SimpleConv]') 
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')
    self.parser.add_argument('--num_head_conv', type=int, default=1,
                             help='number of conv layers before each output head')
    self.parser.add_argument('--head_kernel', type=int, default=3, help='')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')
    # self.parser.add_argument('--not_idaup', action='store_true')
    self.parser.add_argument('--num_classes', type=int, default=-1)
    self.parser.add_argument('--num_resnet_layers', type=int, default=101)
    self.parser.add_argument('--backbone', default='dla34',
                             help='backbone for the generic detection network')
    self.parser.add_argument('--neck', default='dlaup',
                             help='neck for the generic detection network')
    self.parser.add_argument('--msra_outchannel', type=int, default=256)
    # self.parser.add_argument('--efficient_level', type=int, default=0)
    self.parser.add_argument('--prior_bias', type=float, default=-4.6, # -2.19 
                            help='prior bias for last output layer in heatmap head')
    self.parser.add_argument('--extended_head_arch', action='store_true',
                            help='Set architecture of secondary heads to CenterFusion'
                                 'theory (3x3 -> 3x3 -> 3x3 -> 1x1). Deviates from the'
                                 'CenterFusion implementation (1x1 -> 1x1 -> 1x1 -> 1x1).')                       
    self.parser.add_argument('--bn_in_head_arch', action='store_true',
                            help='Use batchnorm layers in all heads and LFANet in between'
                                 'hidden layers.') 

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    self.parser.add_argument('--dataset_version', default='')
    self.parser.add_argument('--blackin', type=float, default=0.0,
                             help='blackin likeliness for RGB picture.')

    # train
    self.parser.add_argument('--optim', default='adam')
    self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--lr_step', type=str, default='60',
                             help='drop learning rate by a factor of lr_step_factor.')
    self.parser.add_argument('--lr_step_factor', type=str, default='1e-1',
                             help='drop learning rate by this factor at lr_step.')                      
    self.parser.add_argument('--momentum', type=float, default='0.9',
                             help='Momentum for SGD optimizer.')
    self.parser.add_argument('--weight_decay', type=float, default='1e-4',
                             help='Weight decay for optimizer.')
    self.parser.add_argument('--save_point', type=str, default='90',
                             help='when to save the model to disk.')
    self.parser.add_argument('--num_epochs', type=int, default=70,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=10,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')
    self.parser.add_argument('--ltrb', action='store_true',
                             help='left top right bottom Bounding box.')
    self.parser.add_argument('--ltrb_weight', type=float, default=0.1,
                             help='')
    self.parser.add_argument('--reset_hm', action='store_true')
    self.parser.add_argument('--reuse_hm', action='store_true')
    # self.parser.add_argument('--use_kpt_center', action='store_true')
    # self.parser.add_argument('--add_05', action='store_true')
    self.parser.add_argument('--dense_reg', type=int, default=1, help='')
    self.parser.add_argument('--shuffle_train', action='store_true',
                             help='shuffle training dataloader')
    
    # test
    self.parser.add_argument('--flip_test', action='store_true',
                             help='flip data augmentation.')
    self.parser.add_argument('--test_scales', type=str, default='1',
                             help='multi scale test augmentation.')
    self.parser.add_argument('--nms', action='store_true',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=100,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_short', type=int, default=-1)
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')
    # self.parser.add_argument('--map_argoverse_id', action='store_true',
    #                          help='if trained on nuscenes and eval on kitti')
    self.parser.add_argument('--out_thresh', type=float, default=-1,
                             help='')
    self.parser.add_argument('--depth_scale', type=float, default=1,
                             help='')
    self.parser.add_argument('--save_results', action='store_true')
    self.parser.add_argument('--load_results', default='')
    self.parser.add_argument('--use_loaded_results', action='store_true')
    self.parser.add_argument('--ignore_loaded_cats', default='')
    self.parser.add_argument('--model_output_list', action='store_true',
                             help='Used when convert to onnx')
    self.parser.add_argument('--non_block_test', action='store_true')
    self.parser.add_argument('--vis_gt_bev', default='',
                             help='path to gt bev images')
    self.parser.add_argument('--kitti_split', default='3dop',
                             help='different validation split for kitti: '
                                  '3dop | subcnn')
    self.parser.add_argument('--test_focal_length', type=int, default=-1)

    # dataset
    self.parser.add_argument('--rand_crop', action='store_true',
                             help='Use the random crop data augmentation'
                                  'from CornerNet.')
    self.parser.add_argument('--not_max_crop', action='store_true',
                             help='used when the training dataset has'
                                  'inbalanced aspect ratios.')
    self.parser.add_argument('--shift', type=float, default=0.1,
                             help='when not using random crop, 0.1'
                                  'apply shift augmentation.')
    self.parser.add_argument('--scale', type=float, default=0,
                             help='when not using random crop, 0.4'
                                  'apply scale augmentation.')
    self.parser.add_argument('--aug_rot', type=float, default=0, 
                             help='probability of applying '
                                  'rotation augmentation.')
    self.parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    self.parser.add_argument('--flip', type=float, default=0.5,
                             help='probability of applying flip augmentation.')
    self.parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')

    # Tracking
    self.parser.add_argument('--tracking', action='store_true')
    self.parser.add_argument('--pre_hm', action='store_true')
    self.parser.add_argument('--same_aug_pre', action='store_true')
    self.parser.add_argument('--zero_pre_hm', action='store_true')
    self.parser.add_argument('--hm_disturb', type=float, default=0)
    self.parser.add_argument('--lost_disturb', type=float, default=0)
    self.parser.add_argument('--fp_disturb', type=float, default=0)
    self.parser.add_argument('--pre_thresh', type=float, default=-1)
    self.parser.add_argument('--track_thresh', type=float, default=0.3)
    self.parser.add_argument('--new_thresh', type=float, default=0.3)
    self.parser.add_argument('--max_frame_dist', type=int, default=3)
    self.parser.add_argument('--ltrb_amodel', action='store_true')
    self.parser.add_argument('--ltrb_amodel_weight', type=float, default=0.1)
    self.parser.add_argument('--public_det', action='store_true')
    self.parser.add_argument('--no_pre_img', action='store_true')
    self.parser.add_argument('--zero_tracking', action='store_true')
    self.parser.add_argument('--hungarian', action='store_true')
    self.parser.add_argument('--max_age', type=int, default=-1)


    # loss
    self.parser.add_argument('--tracking_weight', type=float, default=1)
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for peak heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for peak local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    self.parser.add_argument('--hp_weight', type=float, default=1,
                             help='loss weight for human pose offset.')
    self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                             help='loss weight for human peak heatmap.')
    self.parser.add_argument('--amodel_offset_weight', type=float, default=1,
                             help='Please forgive the typo.')
    self.parser.add_argument('--dep_weight', type=float, default=1,
                             help='loss weight for depth.')
    self.parser.add_argument('--dep_res_weight', type=float, default=1,
                             help='loss weight for depth residual.')
    self.parser.add_argument('--dim_weight', type=float, default=1,
                             help='loss weight for 3d bounding box size / dimensions.')
    self.parser.add_argument('--rot_weight', type=float, default=1,
                             help='loss weight for orientation.')
    self.parser.add_argument('--nuscenes_att', action='store_true',
                             help='Use head for nuscenes attributes explicitly.')
    self.parser.add_argument('--nuscenes_att_weight', type=float, default=1,
                             help='loss weight for nuscenes attributes.')
    self.parser.add_argument('--velocity', action='store_true',
                             help='Use head for velocity explicitly.')
    self.parser.add_argument('--velocity_weight', type=float, default=1,
                             help='loss weight for velocity.')
    self.parser.add_argument('--pc_lfa_feat_weight', type=float, default=1,
                             help='loss weight for LFANet.')
    # self.parser.add_argument('--custom_rotbin_loss', action='store_true')
    self.parser.add_argument('--use_sec_heads', action='store_true',
                             help='Use all secondary heads which are the heads'
                                  'for velocity, nuscenes attributes, secondary'
                                  'depth and secondary rotation. If velocity'
                                  'and nuscenes attributes are not used'
                                  'explicitly they are still used if this flag'
                                  'is activated.')
    self.parser.add_argument('--no_lfa_val_loss', action='store_true',
                             help='Deactivate loss calculation for LFANet. Saves time.')
                              
    # custom dataset
    self.parser.add_argument('--custom_dataset_img_path', default='')
    self.parser.add_argument('--custom_dataset_ann_path', default='')

    # point clouds and nuScenes dataset
    self.parser.add_argument('--pointcloud', action='store_true')
    self.parser.add_argument('--train_split', default='train',
                             choices=['train','mini_train', 'train_detect', 'train_track', 'mini_train_2', 'trainval', 'tiny_train',\
                                      'wee_train', 'nano_train', 'debug_train', 'night_and_rain_train', 'night_rain_train'])
    self.parser.add_argument('--val_split', default='val',
                             choices=['val','mini_val','test', 'tiny_val', 'wee_val', 'nano_val', 'debug_val', 'night_and_rain_val', 'night_rain_val','night_val'])
    self.parser.add_argument('--max_pc', type=int, default=1000,
                             help='maximum number of points in the point cloud')
    self.parser.add_argument('--r_a', type=float, default=250,
                             help='alpha parameter for hm size calculation')
    self.parser.add_argument('--r_b', type=float, default=5,
                             help='beta parameter for hm size calculation')
    self.parser.add_argument('--img_format', default='jpg',
                             help='debug image format')
    self.parser.add_argument('--max_pc_depth', type=float, default=60.0,
                             help='remove points with greater depth value than max_pc_depth meters')
    self.parser.add_argument('--freeze_layers', action='store_true',
                             help='Freeze the backbone network. Additional flags freeze other network parts.')
    self.parser.add_argument('--freeze_lfa', action='store_true', help='Freeze LFANet additionally to backbone.')
    self.parser.add_argument('--freeze_prim', action='store_true', help='Freeze primary heads additionally to backbone.')
    self.parser.add_argument('--freeze_sec', action='store_true', help='Freeze secondary heads additionally to backbone.')
    self.parser.add_argument('--set_eval_layers', action='store_true', help='NOTE: This is needed when training LFANet standalone! Sets frozen layers to eval mode of torch. This only modifies the behavior of BN and Dropout layers. For BN layers they dont track batch parameters anymore and Dropout layers dont dropout.')
    self.parser.add_argument('--radar_sweeps', type=int, default=3,
                             help='number of radar sweeps in point cloud')
    self.parser.add_argument('--warm_start_weights', action='store_true',
                             help='try to reuse weights even if dimensions dont match')
    self.parser.add_argument('--pc_z_offset', type=float, default=0,
                             help='raise all Radar points in z direction')
    self.parser.add_argument('--eval_n_plots', type=int, default=0,
                             help='number of sample plots drawn in eval')
    self.parser.add_argument('--eval_render_curves', action='store_true',
                             help='render and save evaluation curves')
    self.parser.add_argument('--hm_transparency', type=float, default=0.7,
                             help='heatmap visualization transparency')
    self.parser.add_argument('--iou_thresh', type=float, default=0,
                             help='IOU threshold for filtering overlapping detections')
    self.parser.add_argument('--pillar_dims', type=str, default='1.5,0.2,0.2',
                             help='Radar pillar dimensions (h,w,l)')
    # self.parser.add_argument('--show_velocity', action='store_true')
    self.parser.add_argument('--save_model_graph', action='store_true',
                             help='Save the model graph in tensorboard.')    
    self.parser.add_argument('--annotation_dir_ending', type=str, default='',
                             help='Change this when using different annotation folder dir, e.g. <-->_OVA_RF -> "_OVA_RF"')
    self.parser.add_argument('--rcs_feature_hm', action='store_true',
                             help='Add RCS value of radar sensor as additional radar feature heatmap.')
    
    # Frustum
    self.parser.add_argument('--use_dist_for_frustum', action='store_true',
                             help='Instead of the Frustum based on depth, use distance to the radar points instead.')
    self.parser.add_argument('--frustumExpansionRatioVal', type=float, default=0.5,
                             help='Frustum size threshold gets multiplied by (1+frustumExpansionRatioVal) in Validation. Called delta in Paper')
    self.parser.add_argument('--frustumExpansionRatioTrain', type=float, default=0.5,
                             help='Frustum size threshold gets multiplied by (1+frustumExpansionRatioTrain) in Training')
    # Snapshot (LFANet) 
    self.parser.add_argument('--snap_resolution', type=int, default=32,
                            help='Square resolution [res,res] of the snapshot taken'
                                 'from the frustum. Every snapshot is scaled to this'
                                 'resolution. To work with the LFANet it has to be a'
                                 'power of 2.')
    self.parser.add_argument('--snap_channels', type=str, default="z,vx_comp,vz_comp,rcs,dts,cat",
                            help='Channels the snapshot has.\n'
                                 'By default:\n'
                                 '\t- z\n'
                                 '\t- vx_comp\n'
                                 '\t- vz_comp\n'
                                 '\t- RCS\n'
                                 '\t- Delta ts\n'
                                 '\t- category')
    self.parser.add_argument('--snap_method', default='proj',
                            help='Sets the snap generation method. Options are'
                                 'BEV [Birds Eye View Frustum calc]| proj [Project similar to Nabati]') 
    self.parser.add_argument('--lfa_pillar_size', type=float, default=0.0,
                             help='Size of pillars in LFANet in meters. NOTE!: Only used if lfa_pillar_pixel == 0')
    self.parser.add_argument('--lfa_pillar_pixel', type=int, default=3,
                             help='Use pixel in LFA Association instead of absolute size the size in [pixels]. \
                                   When set to >0, pillars are displayed in pixels.')                             
    # self.parser.add_argument('--sort_det_by_depth', action='store_true',
    #                          help='Sort detections by distance in creation of pc BB hm.')
    # self.parser.add_argument('--normalize_depth_input_lfa', action='store_true',
    #                          help='Normalize depth channel in snap as the input into LFANet.')
    self.parser.add_argument('--normalize_cat_input_lfa', action='store_true',
                             help='Normalize category channel in snap as the input into LFANet.')
    
    
    # LFANet Architecture
    self.parser.add_argument('--use_lfa', action='store_true',
                             help='Use learned frustum association.')
    self.parser.add_argument('--train_with_gt', action='store_true', help='While training instead of filling the output of LFAnet to the pc_box_hm, use the feats of the annotation. Equal to "eval_with_gt" just for training.')
    self.parser.add_argument('--eval_with_gt', action='store_true', help='While validating use feats of annotations in pc_box_hm instead of output of LFANet. Equal to "train_with_gt" just for validation.')
    self.parser.add_argument('--lfa_with_ann', action='store_true',
                             help='Input annotations in LFANet in training. This is to judge the performance of LFANet standalone without the'
                                  'influence of CenterFusion.')
    self.parser.add_argument('--lfa_forward_to_sec', action='store_true', help='Pairable with "lfa_with_ann". Use LFA to fill in pc_box_hm instead of feats of ann.')
    self.parser.add_argument('--lfa_skip_loss', action='store_true', help='To train LFA too but indirect from head loss and not from LFA loss. Only makes sense when LFA not frozen.')
    self.parser.add_argument('--lfa_network', type=str, default='img_global',
                             help='Choose network type of LFANet.',
                             choices=['img', 'img_dense', 'img_global', 'img_global_avg',
                                      'pc', 'pc_dense', 'pc_dilated', 'pc_dilated_dense'])
    self.parser.add_argument('--num_lfa_filters', type=str, default='256,256,256,256',
                             help='Number of filters in convolutional layers in LFANet.'
                                  'Can be multiple ints to simutaneosly set the number of layers in the global variant.'
                                  'For other types a single int is required that is used for all layers. The number of layers is predefined for this case.')
    self.parser.add_argument('--increase_nr_channels', action='store_true',
                             help='Increase the number of filters/channels per layer'
                                  'proportionally to the downsizing factor.')
    self.parser.add_argument('--num_lfa_fc_layers', type=int, default=3, 
                             help='Total number of fully connected layers at the end of LFA.'
                                  'If chosen to be 0: Use conv layer wíth 1x1 kernel instead.')
    self.parser.add_argument('--lfa_weights', type=str, default='1,1,1',
                             help='Weights for errors of features in pc_feat_lvl.'
                                   'It has to have the same length as pc_feat_lvl!')
    self.parser.add_argument('--use_pointnet', action='store_true', help='Use PointNet in LFANet [NOT IMPLEMENTED YET]')
    self.parser.add_argument('--use_rules_based', action='store_true',
                             help='Use rules based Frustum association.')
    self.parser.add_argument('--not_use_dcn', action='store_true',
                             help='Don\'t use DCN layers but standard torch Conv2D layers in LFANet.')
    self.parser.add_argument('--lfa_match_channels', action='store_true',
                             help='Add a convolutional layer that maps the radar feature channels to the channels of the image-based feature map.')

    # Post LFANet
    self.parser.add_argument('--limit_frustum_points', type=int, default=0,
                            help='Minimal amount of frustum points such that LFANet'
                                 'makes a prediction for the frustum. The idea is to'
                                 'skip the LFANet (and its training) if there are not'
                                 'enough points to reasonably predict accurately.')
    self.parser.add_argument('--limit_use_closest', action='store_true',
                             help='Uses the features of the closest point as in Nabati when not enough points are inside the frustum for LFANet. The limit for LFANet is given in limit_frustum_points')
    self.parser.add_argument('--limit_use_vel', action='store_true',
                             help='Use only the velocity of the closest point as in Nabati when not enough points are inside the frustum for LFANet. The limit for LFANet is given in limit_frustum_points')
    self.parser.add_argument('--lfa_pred_thresh', type=float, default=0.0,
                             help='Threshold for certainty score of prediction. LFANet is skipped in training (with pred of prim) and validation. Set to greater than zero to save runtime since LFANet is skipped for these predicitions.')
    self.parser.add_argument('--lfa_not_use_amodel', action='store_true',
                             help='Not use amodel bounding box to fill in the pc bb hm.')
    self.parser.add_argument('--hm_to_box_ratio', type=float, default=0.3,
                             help='Ratio of 2D bbox to box that is filled with radar data in pc_box_hm.')
    self.parser.add_argument('--bound_lfa_to_frustum', action='store_true',
                             help='Bound the output of LFANet to the Frustum boundaries.')
    self.parser.add_argument('--lfa_proj_vel_to_rad', action='store_true', 
                             help='Engineering solution to improve performance of LFANet since pred velocity always needs to be radial -> restrict degree of freedom by projecting the predicted absolute value onto the radial axis.')     

    # Early Fusion
    self.parser.add_argument('--use_early_fusion', action='store_true',
                             help='Activate early fusion.')
    self.parser.add_argument('--early_fusion_channels', type=str, default="z,vx_comp,vz_comp,rcs",
                            help='Channels used as additional layers in early fusion.\n'
                                 'By default:\n'
                                 '\t- z\n'
                                 '\t- compensated v_x\n'
                                 '\t- compensated v_z\n'
                                 '\t- RCS')
    self.parser.add_argument('--early_fusion_projection_height', type=float, default=3.0,
                              help='Size of the 2D lines in the early fusion feature images in [m].')
    self.parser.add_argument('--early_fusion_pixel_width', type=float, default=6,
                              help='Width of the 2D lines in the early fusion feature images in pixels. \
                              This width is given in the dimensions of the original sized image (1600x900)')

    # Merge two models 
    self.parser.add_argument('--load_model2', default='',
                            help='path to pretrained model')
    self.parser.add_argument('--merge_models', action='store_true',
                             help='Take backbone, prim, sec heads from "load_model"'
                                  'and LFANet from "load_model2". Only merge them and'
                                  'skip training and val')
                            
  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)
  
    if opt.test_dataset == '':
      opt.test_dataset = opt.dataset
    
    # System setup
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.lr_step_factor = float(opt.lr_step_factor) 
    opt.save_point = [int(i) for i in opt.save_point.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]
    opt.save_imgs = [i for i in opt.save_imgs.split(',')] \
      if opt.save_imgs != '' else []
    opt.ignore_loaded_cats = \
      [int(i) for i in opt.ignore_loaded_cats.split(',')] \
      if opt.ignore_loaded_cats != '' else []
    opt.num_workers = max(opt.num_workers, 2 * len(opt.gpus)) # at least 2 workers per gpu
    if opt.debug > 0 or opt.eval_frustum > 0:
      # Set the number of workers to 0, otherwise the debugging/evaluation does not work
      opt.num_workers = 0
  
    opt.pre_img = False
    if 'tracking' in opt.task:
      print('Running tracking')
      opt.tracking = True
      opt.out_thresh = max(opt.track_thresh, opt.out_thresh)
      opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
      opt.new_thresh = max(opt.track_thresh, opt.new_thresh)
      opt.pre_img = not opt.no_pre_img
      print('Using tracking threshold for out threshold!', opt.track_thresh)
      if 'ddd' in opt.task:
        opt.show_track_color = True

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

    # Different channel numbers depending on backbone
    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64

    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1 

    # Batch sizes
    if opt.master_batch_size == -1:
      if opt.batch_size < len(opt.gpus):
        opt.master_batch_size = 1
      else:      
        opt.master_batch_size = opt.batch_size // len(opt.gpus)   # default: evenly distribute batches over gpus
    rest_batch_size = (opt.batch_size - opt.master_batch_size)  # = batch_size mod nr_gpus
    opt.chunk_sizes = [opt.master_batch_size]       # filled with [master_chunk, slave1_chunk, slave2_chunk,...]
    for i in range(len(opt.gpus) - 1):              # if there are 2 or more gpus all above the first are slaves and handle the rest in the batch
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1) # distribute rest evenly over all but master
      if i < rest_batch_size % (len(opt.gpus) - 1): # there is a rest that can be distributed again
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('Training chunk_sizes (master, slave1, slave2, ...): ', opt.chunk_sizes)

    # In debug reset all to default (can be different default than for normal execution)
    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]    # only one gpu
      opt.master_batch_size = -1  

    # Log dirs
    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    opt.log_dir = opt.save_dir + '/logs_{}'.format(time_str)
    while os.path.isdir(opt.log_dir):
      opt.log_dir += '_I'
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    
    # For experimenting
    if opt.resume and opt.load_model == '':
      opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')

    # Point cloud settings
    opt.pc_atts = ['x', 'y', 'z', 'dyn_prop', 'id', 'rcs', 'vx', 'vz', 
                    'vx_comp', 'vz_comp', 'is_quality_valid', 
                    'ambig_state', 'x_rms', 'z_rms', 'invalid_state', 
                    'pdh0', 'vx_rms', 'vz_rms', 'dts']
    opt.pc_attr_ind = {x:i for i,x in enumerate(opt.pc_atts)}
    opt.pillar_dims = [float(i) for i in opt.pillar_dims.split(',')]
    
    # Glossary: hm = heatmap, dep = depth, sec = secondary (head), cat = category, pc = point cloud
    opt.hm_dist_thresh = None
    opt.sigmoid_dep_sec = False
    opt.secondary_heads = []
    opt.sec_head_convs = {} # Heads of sec network
    opt.normalize_depth = False
    opt.normalize_depth_input_lfa = False
    # Sort detections by distance in creation of pc BB hm.
    opt.sort_det_by_depth = False
    opt.disable_frustum = False
    # Use custum rotbin loss (bugfixed)
    opt.custom_rotbin_loss = True  # always True since it showed better results
    # Freeze layers for transfer learning
    opt.layers_to_freeze = [
      'base', 
      'dla_up',
      'ida_up'
    ]
    if opt.freeze_lfa:
      opt.layers_to_freeze.append('lfa')
    if opt.freeze_prim:
      opt.layers_to_freeze.append('hm')
      opt.layers_to_freeze.append('reg')
      opt.layers_to_freeze.append('wh')
      opt.layers_to_freeze.append('rot')
      opt.layers_to_freeze.append('dim')
      opt.layers_to_freeze.append('amodel_offset')
      opt.layers_to_freeze.append('dep')
    
    if opt.freeze_sec:
      opt.layers_to_freeze.append('dep_sec')
      opt.layers_to_freeze.append('rot_sec')
      opt.layers_to_freeze.append('veloctiy')
      opt.layers_to_freeze.append('nuscenes_att')
      
    if opt.freeze_layers and opt.freeze_prim and opt.freeze_sec and not opt.set_eval_layers:
      print('WARNING! Only LFANet is trained while the rest is NOT set to eval. This leads to corrupt learning due to no direct connection of BN layers in backbone and loss while the BN layers still accumulate batch data.')
    opt.layers_to_eval = opt.layers_to_freeze

    if opt.use_rules_based and opt.use_lfa:
      assert ValueError ('Both, LFANet and rules based Frustum Association activated.')

    if opt.pointcloud:
      ##------------------------------------------------------------------------
      opt.pc_roi_method = "pillars" # alternative: "hm"
      opt.pc_feat_lvl = [
        'pc_dep', # depth
        'pc_vx',  # velocity in x
        'pc_vz',  # velocity in z
      ]              
          # 'pc_dts', # delta timestamp

      if opt.rcs_feature_hm:
        # Append RCS to list of pointcloud feature levels when corresponding option is true
        opt.pc_feat_lvl.append('pc_rcs')
      
      opt.snap_channels = opt.snap_channels.split(',')
      if 'z' in opt.snap_channels:
        opt.lfa_index_z = opt.snap_channels.index('z') # index of depth channel
      if 'dts' in opt.snap_channels:
        opt.lfa_index_dts = opt.snap_channels.index('dts') # index of time step channel
            
      opt.lfa_channel_in = len(opt.snap_channels) # channels in snapshot
      opt.lfa_pc_nr_feat = len(opt.pc_feat_lvl)

      opt.early_fusion_channels = opt.early_fusion_channels.split(',')
      opt.early_fusion_channel_indices = [opt.pc_attr_ind[channel] \
                                          for channel in opt.early_fusion_channels]
      # Calculate factor for dts normalization of radar with 13 Hz
      opt.dts_norm = opt.radar_sweeps/13

      opt.disable_frustum = False
      opt.sigmoid_dep_sec = True  # always True since it showed better results
      opt.normalize_depth = True  # always True since it showed better results
      opt.normalize_depth_input_lfa = True  # always True since it showed better results
      # Sort detections by distance in creation of pc BB hm.
      opt.sort_det_by_depth = True # always True since it showed better results
      if opt.use_sec_heads:
        opt.secondary_heads = ['velocity', 'nuscenes_att', 'dep_sec', 'rot_sec']
      if opt.velocity and 'velocity' not in opt.secondary_heads:
        print('WARNING! Velocity head is used even though not all',
              'secondary heads are used!')
        opt.secondary_heads.append('velocity')
      if opt.nuscenes_att and 'nuscenes_att' not in opt.secondary_heads:
        print('WARNING! Nuscenes attribute head is used even though not all',
              'secondary heads are used!')
        opt.secondary_heads.append('nuscenes_att')
      opt.hm_dist_thresh = {
        'car': 0, 
        'truck': 0,
        'bus': 0,
        'trailer': 0, 
        'construction_vehicle': 0, 
        'pedestrian': 1,
        'motorcycle': 1,
        'bicycle': 1, 
        'traffic_cone': 0, 
        'barrier': 0
      }

      opt.sec_head_convs = {head: 3 for head in opt.secondary_heads}
      
      opt.pc_feat_channels = {feat: i for i,feat in enumerate(opt.pc_feat_lvl)}

      opt.pc_dep_index = opt.pc_feat_channels['pc_dep']
      opt.pc_vx_index = opt.pc_feat_channels['pc_vx']
      opt.pc_vz_index = opt.pc_feat_channels['pc_vz']

    # RGB channels as input to backbone
    opt.num_img_channels = {'cam': 3}

    if opt.use_early_fusion:
      # Add channels from radar to backbone
      opt.num_img_channels['radar'] = len(opt.early_fusion_channels)
    

    if opt.use_lfa:
      # Weight different errors of pc features in LFANet
      weights = [int(s) for s in opt.lfa_weights.split(',')]
      assert len(weights) == len(opt.pc_feat_lvl), "There have to be as many weights as pc features."
      for feat in opt.pc_feat_lvl:
        print("Weight for ", feat, " error is: ", weights[opt.pc_feat_channels[feat]])
      opt.lfa_weights = torch.tensor(weights)#

      # LFA Filters can be either a scalar or a list
      opt.num_lfa_filters = [int(i) for i in opt.num_lfa_filters.split(',')]
      if len(opt.num_lfa_filters) == 1:
        opt.num_lfa_filters = opt.num_lfa_filters[0] # turn to scalar

      # LFA limit
      if opt.limit_use_closest and opt.limit_use_vel:
        print('WARNING: Mode of frustum point limit is ambigous. Velocity mode dominates.')
    
    if opt.merge_models:
      if opt.load_model == '' or opt.load_model2 == '':
        raise ValueError("Not enough models given to merge.")

    CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
    CAT_IDS = {v: i for i, v in enumerate(CATS)}
    
    opt.cat_norm = len(CATS) // 2 - 0.5 # = 4.5 for nuScenes [not 5 to prevent class id to be zero for obejcts since 0 corresponds to background]

    # Assign dist threshold to index of category and not category itself 
    if opt.hm_dist_thresh is not None: # i.e. if opt.pointcloud
      temp = {}
      for (k,v) in opt.hm_dist_thresh.items():
        temp[CAT_IDS[k]] = v
      opt.hm_dist_thresh = temp
    
    # Debug
    opt.show_velocity = True # Always useful
    
    return opt


  def update_dataset_info_and_set_heads(self, opt, dataset):
    opt.num_classes = dataset.num_categories \
                      if opt.num_classes < 0 else opt.num_classes
    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h, input_w = dataset.default_resolution
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
  
    # Configure heads, i.e. set up the number of 'classes' which need to be regressed for each category/head
    opt.heads = {'hm': opt.num_classes, 'reg': 2, 'wh': 2}

    if 'tracking' in opt.task:
      opt.heads.update({'tracking': 2})

    if 'ddd' in opt.task:
      opt.heads.update({'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2})

    if 'multi_pose' in opt.task:
      opt.heads.update({
        'hps': dataset.num_joints * 2, 'hm_hp': dataset.num_joints,
        'hp_offset': 2})
        
    # ltrb_amodel is to use the left, top, right, bottom bounding box representation 
    # to enable detecting out-of-image bounding box (important for MOT datasets)
    if opt.ltrb:
      opt.heads.update({'ltrb': 4})
    if opt.ltrb_amodel:
      opt.heads.update({'ltrb_amodel': 4})

    # Secondary heads
    if 'nuscenes_att' in opt.secondary_heads:
      opt.heads.update({'nuscenes_att': 8})
    if 'velocity' in opt.secondary_heads:
      opt.heads.update({'velocity': 3})
    if 'dep_sec' in opt.secondary_heads:
      opt.heads.update({'dep_sec': 1})
    if 'rot_sec' in opt.secondary_heads:
      opt.heads.update({'rot_sec': 8})

    # Append vel and attr heads as prim heads if wanted to do so
    if opt.velocity and 'velocity' not in opt.secondary_heads:
      print('WARNING! Velocity head is used as a primary head!')
      opt.heads.update({'velocity': 3})
    if opt.nuscenes_att and 'nuscenes_att' not in opt.secondary_heads:
      print('WARNING! Nuscenes attribute head is used as a primary head!')
      opt.heads.update({'nuscenes_att': 8})

    weight_dict = {'hm': opt.hm_weight, 'wh': opt.wh_weight,
                   'reg': opt.off_weight, 'hps': opt.hp_weight,
                   'hm_hp': opt.hm_hp_weight, 'hp_offset': opt.off_weight,
                   'dep': opt.dep_weight, 'dep_res': opt.dep_res_weight,
                   'rot': opt.rot_weight, 'dep_sec': opt.dep_weight,
                   'dim': opt.dim_weight, 'rot_sec': opt.rot_weight,
                   'amodel_offset': opt.amodel_offset_weight,
                   'ltrb': opt.ltrb_weight,
                   'ltrb_amodel': opt.ltrb_amodel_weight,
                   'tracking': opt.tracking_weight,
                   'nuscenes_att': opt.nuscenes_att_weight,
                   'velocity': opt.velocity_weight}
    opt.weights = {head: weight_dict[head] for head in opt.heads}
    
    if opt.use_lfa: 
      opt.weights.update({'pc_lfa_feat': opt.pc_lfa_feat_weight})
       # Check whether pillar size is odd, +1 if not
      if opt.lfa_pillar_pixel > 0 and opt.lfa_pillar_pixel % 2 == 0:
        print(f'{5*"!"} Pillar pixel size is set to even number ({opt.lfa_pillar_pixel}). Changed to ({opt.lfa_pillar_pixel + 1}) {5*"!"}')
        opt.lfa_pillar_pixel += 1
    
    for head in opt.weights:
      if opt.weights[head] == 0:
        del opt.heads[head]   # delete heads with no weighting

    temp_head_conv = opt.head_conv 
    ## Set num_head_conv layers with head_conv channels per head 
    opt.head_conv = {head: [opt.head_conv \
      for i in range(opt.num_head_conv if head != 'reg' else 1)] for head in opt.heads} # keep nr of conv lays before regression head to 1
    # Update custom head in secondary network and add layers
    if opt.pointcloud:
      temp = {k: [temp_head_conv for i in range(v)] for k,v in opt.sec_head_convs.items()}
      opt.head_conv.update(temp)
    
    print('Input height:', opt.input_h, ' & width: ', opt.input_w)
    print('Heads: ', opt.heads)
    print('Weights: ', opt.weights)
    print('Convolutional setup per head: ', opt.head_conv)
    
    # Store max number of objs in opt
    opt.max_objs = dataset.max_objs

    return opt

  def init(self, args=''):
    # only used in demo
    default_dataset_info = {
      'ctdet': 'coco', 'multi_pose': 'coco_hp', 'ddd': 'nuscenes',
      'tracking,ctdet': 'coco', 'tracking,multi_pose': 'coco_hp', 
      'tracking,ddd': 'nuscenes'
    }
    opt = self.parse()
    from dataset.dataset_factory import dataset_factory
    train_dataset = default_dataset_info[opt.task] \
      if opt.task in default_dataset_info else 'coco'
    dataset = dataset_factory[train_dataset]
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
