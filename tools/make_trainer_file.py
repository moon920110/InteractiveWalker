import os
import numpy as np
import h5py
import cv2
from scipy.interpolate import interp1d

input_data_path = '../data/trainers/LSE/2023-07-18_16-09-04_testing_carpets/'
output_left_path = os.path.join(input_data_path, 'left_fitted.avi')
output_right_path = os.path.join(input_data_path, 'right_fitted.avi')
output_hdf5_path = os.path.join(input_data_path, 'fitted_data.hdf5')

# frames where you want to cut : use 'find_frame.py' file to find frames
start_frame = 713
end_frame = start_frame + 2600
desired_frame_cnt = end_frame-start_frame+1

for sub_dir in os.listdir(input_data_path):
    d = os.path.join(input_data_path, sub_dir)
    if d.endswith("left_frame.avi"):
        left_video = cv2.VideoCapture(d)
        frame_rate = int(left_video.get(cv2.CAP_PROP_FPS))
        frame_cnt = 0

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_left_video = cv2.VideoWriter(output_left_path, fourcc, frame_rate, (int(left_video.get(3)), int(left_video.get(4))))

        while True:
            ret, frame = left_video.read()
            if not ret:
                break
            if start_frame <= frame_cnt <= end_frame:
                output_left_video.write(frame)
            if frame_cnt > end_frame:
                break
            frame_cnt += 1
        left_video.release()
        output_left_video.release()
        cv2.destroyAllWindows()

    if d.endswith("right_frame.avi"):
        right_video = cv2.VideoCapture(d)
        frame_rate = int(right_video.get(cv2.CAP_PROP_FPS))
        frame_cnt = 0

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_right_video = cv2.VideoWriter(output_right_path, fourcc, frame_rate,
                                            (int(right_video.get(3)), int(right_video.get(4))))

        while True:
            ret, frame = right_video.read()
            if not ret:
                break
            if start_frame <= frame_cnt <= end_frame:
                output_right_video.write(frame)
            if frame_cnt > end_frame:
                break
            frame_cnt += 1
        right_video.release()
        output_right_video.release()
        cv2.destroyAllWindows()

    if d.endswith(".hdf5"):
        with h5py.File(d, 'r') as f:
            # Find the timestamp of the start frame
            start_time = f['camera-left']['frame_timestamp']['time_s'][start_frame]
            end_time = f['camera-left']['frame_timestamp']['time_s'][end_frame]
            time_desired = f['camera-left']['frame_timestamp']['time_s'][start_frame:end_frame + 1].reshape(-1)
            tile_list = ['tactile-carpet-c', 'tactile-carpet-d', 'tactile-carpet-left', 'tactile-carpet-right']
            with h5py.File(output_hdf5_path, 'w') as f2:
                for key in tile_list:
                    for i in range(len(f[key]['tactile_data']['time_s'])):
                        if f[key]['tactile_data']['time_s'][i] >= start_time:
                            start_frame_tactile = i
                            break
                    for i in range(len(f[key]['tactile_data']['time_s'])):
                        if f[key]['tactile_data']['time_s'][i] >= end_time:
                            end_frame_tactile = i
                            break
                    time_original = f[key]['tactile_data']['time_s'][start_frame_tactile:end_frame_tactile + 1].reshape(-1)
                    data = f[key]['tactile_data']['data'][start_frame_tactile:end_frame_tactile + 1, :, :]
                    upsampled_data = np.empty((desired_frame_cnt, data.shape[1], data.shape[2]))
                    for i in range(data.shape[1]):
                        for j in range(data.shape[2]):
                            slice_data = data[:, i, j]
                            interpolator = interp1d(time_original, slice_data, kind='quadratic', fill_value='extrapolate')
                            upsampled_data[:, i, j] = interpolator(time_desired)
                    f2.create_dataset(f'{key}/tactile_data/data', data=upsampled_data)
