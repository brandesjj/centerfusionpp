import numpy as np
import matplotlib.pyplot as plt
import pickle

categories = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
                'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

def load_data(file_name):
  a_file = open(file_name[0], "rb")
  return pickle.load(a_file)

def main(frustum_dicts):

  #################################################################
  # Plot number of radar points per category.
  #################################################################
  points_per_category = dict.fromkeys(categories, np.empty(0))

  for frustum_dict in frustum_dicts:
    points_per_category[frustum_dict['category']] = np.concatenate((points_per_category[frustum_dict['category']], 
                                                                    np.array([frustum_dict['num_points']])))

  fig_pointscat, ax_pointscat = plt.subplots(figsize = (5.9, 4))
  plt.xticks(rotation=40, ha='right')


  # For all non-zero Frustums, calculate mean
  mean_points_per_category_nz = dict.fromkeys(categories)
  for cat in mean_points_per_category_nz:
    points_nz = np.array(points_per_category[cat])[np.nonzero(points_per_category[cat])[0]]
    mean_points_per_category_nz[cat] = np.mean(points_nz)

  ax_pointscat.bar(*zip(*mean_points_per_category_nz), color = 'blue')


if __name__ == '__main__':
  log_name = 'logs_2022-06-02-19-30'
  file_name = f'../../../exp/ddd/centerfusionpp/{log_name}/snapshot_eval.pkl',

  frustum_dicts = load_data(file_name)

  main(frustum_dicts)
