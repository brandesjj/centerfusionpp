
# Third party libraries
from random import sample
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

# Load the dataset
nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)


def render_sample_front_radar_bev(
                            token: str,
                            nsweeps: int = 1,
                            ) -> None:
    """
    Render the RADAR data from the sample in BEV view.
    """
    record = nusc.get('sample', token)

    # Only the radar sensors corresponding to the following names will be plotted
    radars = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']

    # Separate RADAR from LIDAR and vision.
    radar_data = {}
    camera_data = {}
    for channel, data_token in record['data'].items():
        sd_record = nusc.get('sample_data', data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality == 'camera' and channel == 'CAM_FRONT':
            camera_data[channel] = data_token
        elif sensor_modality == 'radar' and channel in radars:
            radar_data[channel] = data_token

    # Create plots. Radar on left side, front camera on right side
    num_radar_plots = 1 if len(radar_data) > 0 else 0
    n = num_radar_plots + len(camera_data) 
    cols = 2

    # Create matplotlib GridSpec to plot radar on left side and CAMERA_FRONT with each of the radars seperately on the right side
    # fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 24))
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3,3,figure=fig)
    ax_all_radar = fig.add_subplot(gs[:,:-1])
    ax_cam0 = fig.add_subplot(gs[0,-1]) # Subplot for left radar in center camera
    ax_cam1 = fig.add_subplot(gs[1,-1]) # Subplot for center radar radar in center camera
    ax_cam2 = fig.add_subplot(gs[2,-1]) # Subplot for right radar radar in center camera
    axes_cam = {}
    axes_cam['RADAR_FRONT'] = ax_cam1
    axes_cam['RADAR_FRONT_LEFT'] = ax_cam0
    axes_cam['RADAR_FRONT_RIGHT'] = ax_cam2

    # Plot radars into a single subplot.
    if len(radar_data) > 0:
        box_vis_level = BoxVisibility.ANY
        for i, (_, sd_token) in enumerate(radar_data.items()):
            nusc.render_sample_data(sd_token, with_anns=i == 0, box_vis_level=box_vis_level, ax=ax_all_radar, nsweeps=nsweeps,
                                    axes_limit = 60, verbose=False)
        ax_all_radar.set_title('Fused RADARs')

    # Plot front camera including the corresponding point cloud of the radar in separate subplot.
    # for (_, sd_token) in camera_data.items():
    cam_token = camera_data['CAM_FRONT']
    for (channel, ax_cam) in axes_cam.items():
        # nusc.render_pointcloud_in_image(cam_token, ax=ax_cam, pointsensor_channel=channel,
        #                                 show_lidarseg=False, verbose=False)

        nusc.explorer.render_pointcloud_in_image(token, pointsensor_channel=channel,
                                                 camera_channel='CAM_FRONT', show_lidarseg=False, ax = ax_cam,
                                                 verbose=False)
    plt.show()




my_scene = nusc.scene[0]

sample_token = my_scene['first_sample_token']

my_sample = nusc.get('sample', sample_token)

# nusc.render_sample(sample_token)

render_sample_front_radar_bev(sample_token)

# from nuscenes.utils.data_classes import RadarPointCloud
# fig2 = plt.figure(constrained_layout=True)
# gs = GridSpec(1,2, figure=fig2)
# ax_l = fig2.add_subplot(gs[0,0])
# ax_r = fig2.add_subplot(gs[0,1])
# RadarPointCloud.disable_filters()
# nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True, verbose=False, ax=ax_l)
# RadarPointCloud.default_filters()
# nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True, verbose=False, ax=ax_r)
# plt.show()

# nusc.render_pointcloud_in_image(sample_token, pointsensor_channel='RADAR_FRONT_LEFT',camera_channel='CAM_FRONT_LEFT')
# nusc.render_pointcloud_in_image(sample_token, pointsensor_channel='RADAR_FRONT',camera_channel='CAM_FRONT')
# nusc.render_pointcloud_in_image(sample_token, pointsensor_channel='RADAR_FRONT_RIGHT',camera_channel='CAM_FRONT_RIGHT')

# Rander scene without sensor data
# nusc.render_scene(my_scene['token'])

# 
# nusc.render_instance

# Adapted the funtion with plt.show() to show immediately 
# nusc.render_egoposes_on_map('singapore-onenorth', scene_tokens=my_scene['token'])

# nusc.render_sample_data()

# nusc.render_scene_channel_lidarseg(my_scene, 'CAMERA_FRONT')

debug = 0



