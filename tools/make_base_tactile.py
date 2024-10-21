from preprocessing_utils import *
from heatmap_from_keypoint3D import heatmap_gen
import matplotlib.pyplot as plt
import copy
from PIL import Image
import pickle
import cv2
import pandas as pd


data_path = '../data/'
hdf5_squat_path = data_path + 'raw_data/' + name + '/squat1/'
hdf5_squat = hdf5_squat_path + [file for file in os.listdir(hdf5_squat_path) if file.endswith('hdf5')][0]

hdf5_lunge_path = data_path + 'raw_data/' + name +'/lunge1/'
hdf5_lunge = hdf5_lunge_path + [file for file in os.listdir(hdf5_lunge_path) if file.endswith('hdf5')][0]

start_time = hdf5_to_np(hdf5_squat, 'camera-right/frame_timestamp/time_s')[20]
end_time = start_time + 3

left_timestamp, right_timestamp = timestamp_matcher(hdf5_squat)
right_timestamp = [x for x in right_timestamp if x >= start_time]
left_timestamp = left_timestamp[len(left_timestamp) - len(right_timestamp):]
right_timestamp = [x for x in right_timestamp if x <= end_time]
left_timestamp = left_timestamp[:len(right_timestamp)]

# load tactile data
left_closest_cam_idx, right_closest_cam_idx, c_closest_cam_idx, d_closest_cam_idx = closest_camera_matcher(
    hdf5_squat, left_timestamp, right_timestamp)

_, lower_left_upsampled_tactile = up_sampling_tactile(hdf5_squat, 'c', c_closest_cam_idx, right_timestamp)
_, lower_right_upsampled_tactile = up_sampling_tactile(hdf5_squat, 'd', d_closest_cam_idx, right_timestamp)


start_time = hdf5_to_np(hdf5_lunge, 'camera-right/frame_timestamp/time_s')[20]
end_time = start_time + 3

left_timestamp, right_timestamp = timestamp_matcher(hdf5_lunge)
right_timestamp = [x for x in right_timestamp if x >= start_time]
left_timestamp = left_timestamp[len(left_timestamp) - len(right_timestamp):]
right_timestamp = [x for x in right_timestamp if x <= end_time]
left_timestamp = left_timestamp[:len(right_timestamp)]

# load tactile data
left_closest_cam_idx, right_closest_cam_idx, c_closest_cam_idx, d_closest_cam_idx = closest_camera_matcher(
    hdf5_lunge, left_timestamp, right_timestamp)

_, upper_left_upsampled_tactile = up_sampling_tactile(hdf5_lunge, 'left', left_closest_cam_idx, right_timestamp)
_, upper_right_upsampled_tactile = up_sampling_tactile(hdf5_lunge, 'right', right_closest_cam_idx, right_timestamp)

temp_tactile = concat_four_tactiles(upper_left_upsampled_tactile,
                                    upper_right_upsampled_tactile,
                                    lower_left_upsampled_tactile,
                                    lower_right_upsampled_tactile)
base_tactile = temp_tactile.mean(axis=0)
