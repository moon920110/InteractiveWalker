import csv
import cv2
import numpy as np
import h5py
import pickle
import skvideo.io
import pandas as pd
import os
import json
import glob
import mediapipe as mp
from camera_utils import *
from natsort import natsorted

def hdf5_to_np(hdf5: str, parameter: str) -> np.array:
    '''
    Read hdf5 and return into np.array.

    Args:
        hdf5: path of hdf5 file
        parameter: file hierarchy
                    camera-left/frame_timestamp/data
                    camera-left/frame_timestamp/time_s
                    camera-right/frame_timestamp/data
                    camera-right/frame_timestamp/time_s
                    tactile-carpet-left/tactile_data/data
                    tactile-carpet-left/tactile_data/time_s
                    tactile-carpet-right/tactile_data/data
                    tactile-carpet-right/tactile_data/time_s

    Returns:
        n1_np: np.array of hdf5 data
    '''
    hf = h5py.File(hdf5, 'r')
    n1 = hf.get(parameter)
    n1_np = np.asarray(n1)

    return n1_np




def timestamp_matcher(hdf5: str, hdf5_path: str, keypoints=None):
    '''
    Read timestamp information of left, right camera. Return filtered camera timestamp for left, right
    1) filter time difference between left and right camera < 0.01
    2) filter time difference between left and right camera < 0.01 and trim based on keypoints length.

    Args:
        hdf5: path of hdf5 file
        keypoints: path of keypoints.csv

    Returns:
        left_video_timestamp, right_video_timestamp: list of timestamps such that diff < 0.01
        left_timestamp, right_timestamp : trimmed list based on keypoints length
    '''

    left_timestamp = hdf5_to_np(hdf5, 'camera-left/frame_timestamp/time_s')
    right_timestamp = hdf5_to_np(hdf5, 'camera-right/frame_timestamp/time_s')

    ''' for video capture'''''''''
    left_video_path = hdf5_path + [file for file in os.listdir(hdf5_path) if file.endswith('left_frame.avi')][0]
    right_video_path = hdf5_path + [file for file in os.listdir(hdf5_path) if file.endswith('right_frame.avi')][0]
    left_video = cv2.VideoCapture(left_video_path)
    right_video = cv2.VideoCapture(right_video_path)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
    ''''''''''''''''''''''''''''''

    left_timestamp_size = left_timestamp.shape[0]
    right_timestamp_size = right_timestamp.shape[0]

    left_timestamp_fix = np.empty(shape=0)
    right_timestamp_fix = np.empty(shape=0)

    left_video_fix = []
    pose_fix = []

    # camera timestamps time diff between left and right < 0.01
    # for i in range(left_timestamp_size):
    for i in range(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))):
        if right_timestamp_fix.size == 0:
            tmp = 0
        else:
            tmp = int(right_timestamp_fix[-1]) + 1

        for j in range(tmp, int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))):
            if np.abs(float(left_timestamp[i]) - float(right_timestamp[j])) < 0.02: # For using 4 tiles
                left_timestamp_fix = np.append(left_timestamp_fix, i)
                right_timestamp_fix = np.append(right_timestamp_fix, j)
                ''' for video capture'''''''''
                left_video.set(cv2.CAP_PROP_POS_FRAMES, i)
                res, frame0 = left_video.read()
                right_video.set(cv2.CAP_PROP_POS_FRAMES, j)
                res, frame1 = right_video.read()


                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                try:
                    frame0.flags.writeable = False
                except:
                    print('hi')
                frame1.flags.writeable = False
                results0 = pose0.process(frame0)
                results1 = pose1.process(frame1)

                # reverse changes
                frame0.flags.writeable = True
                frame1.flags.writeable = True
                frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

                # check for keypoints detection
                frame0_keypoints = []
                if results0.pose_landmarks:
                    for cnt, landmark in enumerate(results0.pose_landmarks.landmark):
                        if cnt not in pose_keypoints: continue  # only save keypoints that are indicated in pose_keypoints
                        pxl_x = landmark.x * frame0.shape[1]
                        pxl_y = landmark.y * frame0.shape[0]
                        pxl_x = int(round(pxl_x))
                        pxl_y = int(round(pxl_y))
                        cv2.circle(frame0, (pxl_x, pxl_y), 3, (0, 0, 255),
                                   -1)  # add keypoint detection points into figure
                        kpts = [pxl_x, pxl_y]
                        frame0_keypoints.append(kpts)
                else:
                    # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
                    frame0_keypoints = [[-1, -1]] * len(pose_keypoints)

                frame1_keypoints = []
                if results1.pose_landmarks:
                    for cnt, landmark in enumerate(results1.pose_landmarks.landmark):
                        if cnt not in pose_keypoints: continue
                        pxl_x = landmark.x * frame1.shape[1]
                        pxl_y = landmark.y * frame1.shape[0]
                        pxl_x = int(round(pxl_x))
                        pxl_y = int(round(pxl_y))
                        cv2.circle(frame1, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                        kpts = [pxl_x, pxl_y]
                        frame1_keypoints.append(kpts)

                else:
                    # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
                    frame1_keypoints = [[-1, -1]] * len(pose_keypoints)

                frame_p3ds = []
                for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
                    if uv1[0] == -1 or uv2[0] == -1:
                        _p3d = [-1, -1, -1]
                    else:
                        _p3d = DLT(P0, P1, uv1, uv2)  # calculate 3d position of keypoint
                    frame_p3ds.append(_p3d)

                '''
                This contains the 3d position of each keypoint in current frame.
                For real time application, this is what you want.
                '''
                frame_p3ds = np.array(frame_p3ds).reshape((12, 3))
                pose_fix.append(frame_p3ds)
                left_video_fix.append(frame0)
                ''''''''''''''''''''''''''''''
                break
            if np.abs(float(left_timestamp[i]) - float(right_timestamp[j])) > 0.2: break

    left_timestamp_fix = left_timestamp_fix.astype(int)
    right_timestamp_fix = right_timestamp_fix.astype(int)
    pose_fix = np.array(pose_fix)
    left_video_fix = np.array(left_video_fix)

    # left_video_timestamp = left_timestamp_fix
    # right_video_timestamp = right_timestamp_fix
    left_video_timestamp = [float(left_timestamp[x]) for x in left_timestamp_fix]
    right_video_timestamp = [float(right_timestamp[x]) for x in right_timestamp_fix]
    # trim again with keypoints length
    if keypoints is not None:
        keypoints = pd.read_csv(keypoints)
        tmp_left_video_timestamp = [float(left_timestamp[x]) for x in left_timestamp_fix]
        left_timestamp = tmp_left_video_timestamp[:len(keypoints)]
        left_timestamp = [float(left_timestamp[x]) for x in range(len(left_timestamp))]

        tmp_right_video_timestamp = [float(right_timestamp[x]) for x in right_timestamp_fix]
        right_timestamp = tmp_right_video_timestamp[:len(keypoints)]
        right_timestamp = [float(right_timestamp[x]) for x in range(len(right_timestamp))]

        return left_timestamp, right_timestamp, pose_fix

    else:
        return left_video_timestamp, right_video_timestamp, left_video_fix, pose_fix



def closest_camera_matcher(hdf5: str, left_camera_timestamps: list, right_camera_timestamps: list, start_time, end_time) -> list:
    '''
    Compare tactile timestamps(left & right), preprocessed camera timestamps(left & right).
    Match each camera timestamps to the closest tactile timestamps.
    Args:
        hdf5: path of hdf5 file
        left_camera_timestamps: list of timestamps
        right_camera_timestamps: list of timestamps

    Returns:
        left_closest_cam_idx: list of the closest camera indices(left)
        right_closest_cam_idx: list of the closest camera indices(right)

    '''
    left_tactile_timestamps = [float(timestamp) for _, timestamp in
                               enumerate(hdf5_to_np(hdf5, 'tactile-carpet-left/tactile_data/time_s'))]

    right_tactile_timestamps = [float(timestamp) for _, timestamp in
                                enumerate(hdf5_to_np(hdf5, 'tactile-carpet-right/tactile_data/time_s'))]

    c_tactile_timestamps = [float(timestamp) for _, timestamp in
                               enumerate(hdf5_to_np(hdf5, 'tactile-carpet-c/tactile_data/time_s'))]

    d_tactile_timestamps = [float(timestamp) for _, timestamp in
                            enumerate(hdf5_to_np(hdf5, 'tactile-carpet-d/tactile_data/time_s'))]

    left_tactile_timestamps = [x for x in left_tactile_timestamps if x >= start_time]
    left_tactile_timestamps = [x for x in left_tactile_timestamps if x <= end_time]
    right_tactile_timestamps = [x for x in right_tactile_timestamps if x >= start_time]
    right_tactile_timestamps = [x for x in right_tactile_timestamps if x <= end_time]
    c_tactile_timestamps = [x for x in c_tactile_timestamps if x >= start_time]
    c_tactile_timestamps = [x for x in c_tactile_timestamps if x <= end_time]
    d_tactile_timestamps = [x for x in d_tactile_timestamps if x >= start_time]
    d_tactile_timestamps = [x for x in d_tactile_timestamps if x <= end_time]

    left_closest_cam_idx = []
    for tactile_timestamp in left_tactile_timestamps:
        time_differences = []
        for camera_timestamp in right_camera_timestamps:
            time_diff = abs(tactile_timestamp - camera_timestamp)
            time_differences.append(time_diff)

        if min(time_differences) < 1:
            left_closest_cam_idx.append(time_differences.index(min(time_differences)))
        else:
            left_closest_cam_idx.append('NA')
    left_closest_cam_idx = [idx for idx in left_closest_cam_idx if idx != 'NA']

    right_closest_cam_idx = []
    for tactile_timestamp in right_tactile_timestamps:
        time_differences = []
        for camera_timestamp in right_camera_timestamps:
            time_diff = abs(tactile_timestamp - camera_timestamp)
            time_differences.append(time_diff)

        if min(time_differences) < 1:
            right_closest_cam_idx.append(time_differences.index(min(time_differences)))
        else:
            right_closest_cam_idx.append('NA')
    right_closest_cam_idx = [idx for idx in right_closest_cam_idx if idx != 'NA']

    c_closest_cam_idx = []
    for tactile_timestamp in c_tactile_timestamps:
        time_differences = []
        for camera_timestamp in right_camera_timestamps:
            time_diff = abs(tactile_timestamp - camera_timestamp)
            time_differences.append(time_diff)

        if min(time_differences) < 1:
            c_closest_cam_idx.append(time_differences.index(min(time_differences)))
        else:
            c_closest_cam_idx.append('NA')
    c_closest_cam_idx = [idx for idx in c_closest_cam_idx if idx != 'NA']

    d_closest_cam_idx = []
    for tactile_timestamp in d_tactile_timestamps:
        time_differences = []
        for camera_timestamp in right_camera_timestamps:
            time_diff = abs(tactile_timestamp - camera_timestamp)
            time_differences.append(time_diff)

        if min(time_differences) < 1:
            d_closest_cam_idx.append(time_differences.index(min(time_differences)))
        else:
            d_closest_cam_idx.append('NA')
    d_closest_cam_idx = [idx for idx in d_closest_cam_idx if idx != 'NA']

    # if len(left_closest_cam_idx) != len(right_closest_cam_idx):
    #     if len(left_closest_cam_idx) < len(right_closest_cam_idx):
    #         right_closest_cam_idx = right_closest_cam_idx[:len(left_closest_cam_idx)]
    #         print('Length difference!\nleft length < right length')
    #     else:
    #         left_closest_cam_idx = left_closest_cam_idx[:len(right_closest_cam_idx)]
    #         print('Length difference!\nleft length > right length')

    return left_closest_cam_idx, right_closest_cam_idx, c_closest_cam_idx, d_closest_cam_idx


def ratio_generator(start_idx: int, end_idx: int, total_timestamp: list, start_val: np.ndarray, end_val: np.ndarray) -> np.ndarray:
    '''
    Generate data based on ratio of timestamp difference.

    Args:
        start_idx: starting index of target data
        end_idx: ending index of target data
        total_timestamp: list of total timestamps
        start_val: value corresponding to start_idx
        end_val: value corresponding to end_idx

    Return:
        val: generated np.ndarray based on ratio of timestamp difference
    '''
    timestamp = [total_timestamp[idx] for idx in range(start_idx, end_idx + 1)]

    timestamp_diff = np.diff(timestamp)
    total_timestamp_diff = max(timestamp) - min(timestamp)
    timestamp_diff_cumsum = np.cumsum(timestamp_diff)
    val = [start_val] * (len(timestamp_diff) - 1)

    for i in range(len(timestamp_diff)):
        try:
            val[i] = val[i] + (end_val - start_val) * (timestamp_diff_cumsum[i] / total_timestamp_diff)

        except IndexError:
            pass

    return val


def up_sampling_tactile(hdf5: str, carpet_type: str, closest_camera_idx: list, camera_timestamps: list, start_time, end_time):
    '''
    Make key timestamps to apply upsampling, then apply upsampling for tactile data based on ratio of timestamp.

    Args:
        hdf5: path of hdf5 file
        carpet_type: left or right
        closest_camera_idx: list of the closest camera timestamp matched with tactile timestamp
        camera_timestamps: list of timestamps such that filtered by time difference between left and right camera < 0.01 and trimmed based on keypoints length.

    Return:
        key_timestamp_idx: list of timestamp used for applying upsampling
        upsampled_tactile: np.ndarray applied upsampling
    '''

    tactile = hdf5_to_np(hdf5, 'tactile-carpet-' + carpet_type +'/tactile_data/data')
    tactile_timestamps = hdf5_to_np(hdf5, 'tactile-carpet-' + carpet_type + '/tactile_data/time_s')

    tactile_timestamps = [x for x in tactile_timestamps if x <= end_time]
    tactile = tactile[:len(tactile_timestamps)]
    tactile_timestamps = [x for x in tactile_timestamps if x >= start_time]
    tactile = tactile[len(tactile) - len(tactile_timestamps):]
    closest_camera_idx_tactile = list(zip(closest_camera_idx, tactile))

    # closest camera index + 차이 < 1 + 중복 제거 한 camera index,  나중에 upsampling 할 때 기준점이 됨
    key_timestamp_idx = sorted([int(i) for i in np.unique(closest_camera_idx)])

    # Data upsampling을 위해 (전체 카메라 길이, 32, 32) 만들고, 대응되는 데이터가 있는 경우에 대응
    upsampled_tactile = np.zeros((len(camera_timestamps), 32, 32))

    for idx, val in enumerate(key_timestamp_idx):
        for idx_tactile in closest_camera_idx_tactile:
           if idx_tactile[0] == val:
               upsampled_tactile[val] = idx_tactile[1]


    start_end_idx = [(start, end) for start, end in zip(key_timestamp_idx, key_timestamp_idx[1:])]

    # Upsampling
    for idx in range(len(start_end_idx)):
        start_idx = start_end_idx[idx][0]
        end_idx = start_end_idx[idx][1]
        if end_idx - start_idx != 1:
            # generate tactile
            generated_tactile = ratio_generator(start_idx, end_idx, camera_timestamps,
                                            upsampled_tactile[start_idx], upsampled_tactile[end_idx])
            # apply upsampling
            for i in range(start_idx + 1, end_idx):
                upsampled_tactile[i] = generated_tactile[i-start_idx-1]

    # 앞뒤 부분 삭제
    # upsampled_tactile = upsampled_tactile[min(key_timestamp_idx) : max(key_timestamp_idx) + 1]

    return key_timestamp_idx, upsampled_tactile

def up_sampling_tactile_yh(hdf5: str, carpet_type: str, closest_camera_idx: list, camera_timestamps: list):
    '''
    Make key timestamps to apply upsampling, then apply upsampling for tactile data based on ratio of timestamp.

    Args:
        hdf5: path of hdf5 file
        carpet_type: left or right
        closest_camera_idx: list of the closest camera timestamp matched with tactile timestamp
        camera_timestamps: list of timestamps such that filtered by time difference between left and right camera < 0.01 and trimmed based on keypoints length.

    Return:
        key_timestamp_idx: list of timestamp used for applying upsampling
        upsampled_tactile: np.ndarray applied upsampling
    '''
    tactile = hdf5_to_np(hdf5, 'tactile-carpet-' + carpet_type +'/tactile_data/data')
    closest_camera_idx_tactile = list(zip(closest_camera_idx, tactile))

    # closest camera index + 차이 < 1 + 중복 제거 한 camera index,  나중에 upsampling 할 때 기준점이 됨
    key_timestamp_idx = sorted([int(i) for i in np.unique(closest_camera_idx)])

    # Data upsampling을 위해 (전체 카메라 길이, 32, 32) 만들고, 대응되는 데이터가 있는 경우에 대응
    upsampled_tactile = np.zeros((len(camera_timestamps), 32, 32))

    for idx, val in enumerate(key_timestamp_idx):
        for idx_tactile in closest_camera_idx_tactile:
           if idx_tactile[0] == val:
               upsampled_tactile[val] = idx_tactile[1]


    start_end_idx = [(start, end) for start, end in zip(key_timestamp_idx, key_timestamp_idx[1:])]

    # Upsampling
    for idx in range(len(start_end_idx)):
        start_idx = start_end_idx[idx][0]
        end_idx = start_end_idx[idx][1]
        if end_idx - start_idx != 1:
            # generate tactile
            generated_tactile = ratio_generator(start_idx, end_idx, camera_timestamps,
                                            upsampled_tactile[start_idx], upsampled_tactile[end_idx])
            # apply upsampling
            for i in range(start_idx + 1, end_idx):
                upsampled_tactile[i] = generated_tactile[i-start_idx-1]

    # 앞뒤 부분 삭제
    # upsampled_tactile = upsampled_tactile[min(key_timestamp_idx) : max(key_timestamp_idx) + 1]

    return key_timestamp_idx, upsampled_tactile

def rotate_tactile(tactile:np.ndarray, rot:int):
    '''
    Rotate tactile data
    Args:
        tactile: upsampled tactile
        rot: rotation num

    Returns:
        Rotated tactile (32, 32)
    '''
    rotated_tactile = np.zeros((len(tactile), tactile.shape[1], tactile.shape[2]))
    for idx, data in enumerate(tactile):
        data = np.rot90(data, rot)
        rotated_tactile[idx] = data

    return rotated_tactile


def concat_four_tactiles(upper_left_tactile, upper_right_tactile, lower_left_tactile, lower_right_tactile):
    concat_tactiles = np.zeros((len(upper_left_tactile), upper_left_tactile.shape[1]*2, upper_left_tactile.shape[2]*2))
    for idx, tactile in enumerate(zip(upper_left_tactile, upper_right_tactile, lower_left_tactile, lower_right_tactile)):
        upper_left = tactile[0]
        upper_right = tactile[1]
        lower_left = tactile[2]
        lower_right = tactile[3]

        concat_upper = np.concatenate((upper_left, upper_right), axis=1)
        concat_lower = np.concatenate((lower_left, lower_right), axis=1)
        concat_tactile = np.concatenate((concat_lower, concat_upper), axis=0)
        concat_tactiles[idx] = concat_tactile
    return concat_tactiles


def concat_tactiles(left_tactile:np.ndarray, right_tactile:np.ndarray):
    '''
    Concat two tactiles into one, consider flipping and rotation
    Args:
        left_tactile: upsampled left tactile
        right_tactile: upsampled right tactile
        rot: rotation num

    Returns:
        concat_tactile: (32, 64)
    '''
    concat_tactiles = np.zeros((len(left_tactile), left_tactile.shape[1], left_tactile.shape[2] * 2))
    for idx, tactile in enumerate(zip(left_tactile, right_tactile)):
        left = tactile[0]
        right = tactile[1]
        concat_tactile = np.concatenate((left, right), axis=1)
        concat_tactiles[idx] = concat_tactile

    return concat_tactiles

def fitted_keypoints_to_p(keypoints: str, key_timestamp_idx: list, data_path: str) -> None:
    '''
    Fit keypoints with key timestamp and save into pickle format

    Args:
        keypoints: path of keypoints.csv
        key_timestamp_idx: list of timestamp used for applying upsampling
        data_path: path
    '''
    f = open(keypoints, 'r')
    rdr = csv.reader(f)
    data = []
    line_data = []
    for idx, line in enumerate(rdr):
        if idx <= min(key_timestamp_idx) or idx > max(key_timestamp_idx) + 1:
            continue
        line_data = np.asarray(line)
        line_data = np.delete(line_data, [0])
        line_data = np.reshape(line_data, (-1, 3))
        data.append(line_data)
    data = np.asarray(data)
    data = data.astype(float)
    pickle.dump(data, open(data_path + '/' + 'fitted_keypoints.p', "wb"))

    f.close()



def normalize_directly(upsampled_tactile: np.ndarray) -> np.ndarray:
    '''
    Normalize tactile data.

    Args:
        upsampled_tactile: upsampled np.ndarray tactile data

    Return:
        std: normalized tactile data
    '''

    n_min = np.amin(upsampled_tactile)
    n_max = np.amax(upsampled_tactile)
    upsampled_tactile = np.where(upsampled_tactile < n_min, n_min, upsampled_tactile)
    upsampled_tactile = np.where(upsampled_tactile > n_max, n_max, upsampled_tactile)

    std = (upsampled_tactile - n_min)/(n_max - n_min)
    return std

def remove_noise(upsampled_tactile: np.ndarray) -> np.ndarray:
    foot_signal_lower_bound = upsampled_tactile.mean() + 3.0 * upsampled_tactile.std()
    upsampled_tactile[upsampled_tactile < foot_signal_lower_bound] = 0

    return upsampled_tactile

def trim_video(left_video: str, right_video: str, left_video_timestamp: list, right_video_timestamp: list,
               data_path: list, hdf5: list) -> None:
    '''
    Trim video with info of timestamp information.

    Args:
        left_video : data path of original left_video avi
        right_video : data path of original right_video avi
        left_video_timestamp : list of timestamps such that diff < 0.01
        right_video_timestamp : list of timestamps such that diff < 0.01
        data_path : data path to save trimmed video
        hdf5: path of hdf5 file

    Return:
        None
    '''

    left_video_array = np.array(skvideo.io.vread(left_video))
    right_video_array = np.array(skvideo.io.vread(right_video))

    original_left_video_timestamp = hdf5_to_np(hdf5, 'camera-left/frame_timestamp/time_s')
    original_right_video_timestamp = hdf5_to_np(hdf5, 'camera-right/frame_timestamp/time_s')

    left_trim = np.empty(shape=0)   # 잘라버릴 부분들
    right_trim = np.empty(shape=0)  # 잘라버릴 부분들

    # video_left_timestamp = video_left_timestamp.astype(int)
    # video_right_timestamp = video_right_timestamp.astype(int)

    print('Generating video')

    # 왼쪽 자르기
    for i in range(original_left_video_timestamp.shape[0]):
        if i not in left_video_timestamp:
            left_trim = np.append(left_trim, i)
    left_trim = left_trim.astype(int)

    # 오른쪽 자르기
    for j in range(original_right_video_timestamp.shape[0]):
        if j not in right_video_timestamp:
            right_trim = np.append(right_trim, j)
    right_trim = right_trim.astype(int)

    left_video_array = np.delete(left_video_array, left_trim, axis=0)
    right_video_array = np.delete(right_video_array, right_trim, axis=0)
    full_video_array = np.concatenate((left_video_array, right_video_array), axis=2)

    print(left_video_array.shape)
    print(right_video_array.shape)
    print(full_video_array.shape)

    file_name = ('left_fitted_video', 'right_fitted_video', 'full_fitted_video')

    # 비디오 생성
    _, left_height, left_width, _ = left_video_array.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(data_path + '/' + file_name[0] + '.avi', fourcc, 30, (left_width, left_height))
    for i in range(len(left_video_array)):
        left_video_array[i] = cv2.cvtColor(left_video_array[i], cv2.COLOR_BGR2RGB)
        out.write(left_video_array[i])
    out.release()

    _, right_height, right_width, _ = right_video_array.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(data_path + '/' + file_name[1] + '.avi', fourcc, 30, (right_width, right_height))
    for i in range(len(right_video_array)):
        right_video_array[i] = cv2.cvtColor(right_video_array[i], cv2.COLOR_BGR2RGB)
        out.write(right_video_array[i])
    out.release()

    # 전체 비디오 생성
    _, full_height, full_width, _ = full_video_array.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(data_path + '/' + file_name[2] + '.avi', fourcc, 30, (full_width, full_height))
    for i in range(len(full_video_array)):
        full_video_array[i] = cv2.cvtColor(full_video_array[i], cv2.COLOR_BGR2RGB)
        out.write(full_video_array[i])
    out.release()

    print("Generated video")


def p_directory_generator(data_path: str, pfile: str) -> None:
    '''
    Make directories for pickle files

    Args:
        data_path: path
        pfile: pickle file
    '''
    preprocessed_tactile = pd.read_pickle(data_path + '/' + pfile)
    num_directory = len(preprocessed_tactile) // 200 + 1
    for i in range(num_directory):
        try:
            if not os.path.exists(data_path + '/train' + '/' + str(i * 200)):
                os.makedirs(data_path + '/train' + '/' + str(i * 200))
        except OSError as e:
            print(e)

def tmp_p_generator(data_path: str, tactile_p: str, heatmap_p: str, keypoint_p: str) -> None:
    '''
    Generate pickle file without generating folders for every 200
    Args:
        data_path:
        tactile_p:
        heatmap_p:
        keypoint_p:

    Returns:

    '''
    tactile = pd.read_pickle(data_path + '/' + tactile_p)
    heatmap = pd.read_pickle(data_path + '/' + heatmap_p)
    keypoint = pd.read_pickle(data_path + '/' + keypoint_p)

    print('generate tmp train files')
    os.makedirs(data_path + '/tmp_train', exist_ok=True)
    for i, pair in enumerate(zip(tactile, heatmap, keypoint)):
        tmp_path = data_path + '/tmp_train/' + str(i) + '.p'
        with open(tmp_path, 'wb') as f:
            pickle.dump(pair, f)

def trainable_p_generator(data_path: str, tactile_p: str, heatmap_p: str, keypoint_p: str) -> None:
    '''
    Make pickle file consists of tactile, heatmap, and keypoint data with trainable structure

    Args:
        data_path: path
        tactile_p: preprocessed_tactile.p
        heatmap_p: (generated by heatmap_from_keypoint3D.py) fitted_keypoints_heatmap3D.p
        keypoint_p: (generated by heatmap_from_keypoint3D.py) fitted_keypoints_coord.p
    '''
    tactile = pd.read_pickle(data_path + '/' + tactile_p)
    heatmap = pd.read_pickle(data_path + '/' + heatmap_p)
    keypoint = pd.read_pickle(data_path + '/' + keypoint_p)

    print('Making directories')
    p_directory_generator(data_path, tactile_p)

    print('Making folders')
    for directory in os.listdir(data_path + '/train'):
        try:
            for i in range(200):
                tmp_tactile = tactile[int(directory) + i]
                tmp_heatmap = heatmap[int(directory) + i]
                tmp_keypoint = keypoint[int(directory) + i]
                tmp_path = data_path + '/train' + '/' + directory + '/' + str(int(directory) + i) + '.p'
                tmp_p = [tmp_tactile, tmp_heatmap, tmp_keypoint]
                with open(tmp_path, 'wb') as f:
                    pickle.dump(tmp_p, f)
        except IndexError:
            pass

def log_p(target_path:str):
    folders = np.asarray([int(folder) for folder in natsorted(os.listdir(target_path)) if folder != 'log.p'])
    with open(target_path + '/' + 'log.p', 'wb') as f:
        pickle.dump(folders, f)

def to_keypoints3D(path:str) -> None:
    '''
    Args:
        path: str

    Returns: None
    '''

    print(path.split('/')[-1])
    load_dir = path + '/keypoints/keypoints3D/'
    WX = np.genfromtxt (load_dir + 'WX.csv', delimiter=",")
    WY = np.genfromtxt (load_dir + 'WY.csv', delimiter=",")
    WZ = np.genfromtxt (load_dir + 'WZ.csv', delimiter=",")

    WX = WX.transpose()
    WY = WY.transpose()
    WZ = WZ.transpose()

    keypoints_num = WX.shape[1]

    keypoints_3d = np.zeros((WX.shape[0],WX.shape[1]*3))

    for i in range(0, WX.shape[0]):
        for j in range(0, keypoints_num):
            keypoints_3d[i][3*j] = WX[i][j]
            keypoints_3d[i][1+3*j] = WY[i][j]
            keypoints_3d[i][2+3*j] = WZ[i][j]

    keypoints_3d = keypoints_3d[:-10,:]

    colnames = []
    for i in range(0, keypoints_num):
        colnames.append("x"+str(i))
        colnames.append("y"+str(i))
        colnames.append("z"+str(i))

    pd.DataFrame(keypoints_3d).to_csv(path + '/keypoints/keypoints3D.csv', index = True,header=colnames)


def json_to_csv(path:str) -> None:
    '''
    :param path:
    :return: None
    '''
    # experiment_list = os.listdir(path)
    print(path.split('/')[-1])
    left = natsorted(os.listdir(path + '/json/left'))
    right = natsorted(os.listdir(path + '/json/right'))

    frontsub = np.zeros((len(left), 75))
    sidesub = np.zeros((len(right), 75))

    for pair in zip(left, right):
        n = 0
        json_left = json.load(open(path + '/json/left/' + pair[0]))
        json_right = json.load(open(path + '/json/right/' + pair[1]))

        if json_left['people'] != [] and json_right['people'] != []:
            keypoints_left = json_left['people'][0]['pose_keypoints_2d']
            keypoints_right = json_right['people'][0]['pose_keypoints_2d']
            idx = left.index(pair[0])
            frontsub[idx] = keypoints_left
            sidesub[idx] = keypoints_right

        else:
            n += 1
            frontsub.resize((len(left) - n, 75))
            sidesub.resize((len(right) - n, 75))

    csv_dir = path + '/keypoints/keypoints2D/'
    os.makedirs(csv_dir, exist_ok=True)
    pd.DataFrame(frontsub).to_csv(csv_dir + 'frontSub.csv', index=False, header=False)
    pd.DataFrame(sidesub).to_csv(csv_dir + 'sideSub.csv', index=False, header=False)

    os.makedirs(path + '/keypoints/keypoints3D', exist_ok=True)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))

def loadImage(path):
    img = cv2.imread(path)
    cv2.imshow('Test', img)
    cv2.setMouseCallback('Test', onMouse)
    pressedkey = cv2.waitKey(0)
    if pressedkey == 27:
        cv2.destroyAllWindows()

def vertices_checker(path):
    left = glob.glob(path + '/calibration/cam1/image/image2700.jpg')[0] # Select prpoper image // 25
    right = glob.glob(path + '/calibration/cam2/image/image2700.jpg')[0]
    print(path.split('/')[-1])
    print('Click four vertices, then press ESC to change camera\nMake sure keep the ORDER you click')
    print('left')
    loadImage(left)
    print('='*5)
    print('right')
    loadImage(right)

def cam_cap(path):
    '''
    :param path: Calibration video path
    :return: None
    '''
    try:
        cam_name = [file for file in os.listdir(path) if file.split('.')[-1] == 'mkv'][0]
    except IndexError:
        pass
    print('')
    cap = cv2.VideoCapture(path + '/' + cam_name)
    if not os.path.exists(path + '/calibration'):
        os.makedirs(path + '/calibration')
    os.makedirs(path + '/calibration/cam1/image', exist_ok=True)
    os.makedirs(path + '/calibration/cam2/image', exist_ok=True)

    count = 1
    while (cap.isOpened()):
        try:
            ret, frame = cap.read()
            frame1= frame[:,:int(frame.shape[1]/2),:]
            frame2= frame[:,int(frame.shape[1]/2):,:]
            if count % 25 == 0:
                cv2.imwrite(path + '/calibration/cam1/image/image' + str(count) + ".jpg", frame1)
                cv2.imwrite(path + '/calibration/cam2/image/image' + str(count) + ".jpg", frame2)
            count = count + 1
        except AttributeError:
            break
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

def normalize_with_range(data, max, min):
    data = (data-min)/(max-min)
    return data

def softmax(x):
    output = np.exp(x) / np.sum(np.exp(x))
    return output

def make_base_tactile(data_path, name, is_trainer):
    if is_trainer:
        hdf5_squat_path = data_path + 'raw_data/' + name + '/squat2/'
    else:
        hdf5_squat_path = data_path + 'raw_data/' + name + '/squat_video/'
    hdf5_squat = hdf5_squat_path + [file for file in os.listdir(hdf5_squat_path) if file.endswith('hdf5')][0]

    if is_trainer:
        hdf5_lunge_path = data_path + 'raw_data/' + name + '/lunge2/'
    else:
        hdf5_lunge_path = data_path + 'raw_data/' + name + '/lunge_video/'
    hdf5_lunge = hdf5_lunge_path + [file for file in os.listdir(hdf5_lunge_path) if file.endswith('hdf5')][0]

    lower_left_upsampled_tactile = hdf5_to_np(hdf5_squat, 'tactile-carpet-' + 'c' + '/tactile_data/data')
    lower_right_upsampled_tactile = hdf5_to_np(hdf5_squat, 'tactile-carpet-' + 'd' + '/tactile_data/data')
    upper_left_upsampled_tactile = hdf5_to_np(hdf5_lunge, 'tactile-carpet-' + 'left' + '/tactile_data/data')
    upper_right_upsampled_tactile = hdf5_to_np(hdf5_lunge, 'tactile-carpet-' + 'right' + '/tactile_data/data')

    temp_tactile = concat_four_tactiles(upper_left_upsampled_tactile[:50],
                                        upper_right_upsampled_tactile[:50],
                                        lower_left_upsampled_tactile[:50],
                                        lower_right_upsampled_tactile[:50])
    base_tactile = temp_tactile.mean(axis=0)

    return base_tactile

def make_max_emg(data_path, name, isTrainer):
    hdf5_emg_path = data_path + 'raw_data/' + name + '/EMG/'

    if isTrainer:
        column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
                   'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT LUMBAR ES (uV)', 'LT RECT.ABDOM.UP. (uV)',
                   'RT TIB.ANT. (uV)',
                   'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)',
                   'RT LUMBAR ES (uV)',
                   'RT RECT.ABDOM.UP. (uV)', 'Switch (On)']
    else:
        column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
                       'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT RECT.ABDOM.UP. (uV)',
                       'RT TIB.ANT. (uV)',
                       'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)',
                       'RT RECT.ABDOM.UP. (uV)', 'Switch (On)']

    file_list = os.listdir(hdf5_emg_path)
    file_list.remove('info.csv')
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')

    for i, file in enumerate(file_list):
        if i == 0:
            emg_df = pd.read_csv(hdf5_emg_path + file, skiprows=4, names=column_name, low_memory=False)
        else:
            if len(pd.read_csv(hdf5_emg_path + file, skiprows=4, low_memory=False).columns) == 17:
                column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
                               'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT RECT.ABDOM.UP. (uV)',
                               'RT TIB.ANT. (uV)',
                               'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)',
                               'RT RECT.ABDOM.UP. (uV)', 'Switch (On)', 'Switch (On)2']
            df = pd.read_csv(hdf5_emg_path + file, skiprows=4, names=column_name, low_memory=False)
            if name == 'LSJ' and ('crunch' in file):
                df['RT GLUT. MAX. (uV)'] = df['LT GLUT. MAX. (uV)']
            if name == 'PSY' and ('crunch1' in file) :
                df['LT GLUT. MAX. (uV)'] = df['RT GLUT. MAX. (uV)']
            if name == 'YWS' and ('crunch_tactile' in file) :
                df['RT GLUT. MAX. (uV)'] = df['LT GLUT. MAX. (uV)']
            if name == 'JYG' and ('crunch2' in file):
                continue

            if len(pd.read_csv(hdf5_emg_path + file, skiprows=4, low_memory=False).columns) == 17:
                del df['Switch (On)2']
            emg_df = pd.concat([emg_df, df])
    if len(emg_df.columns) == 17:
        del emg_df['Switch (On)2']
    max_emg = emg_df.quantile((0.997))
    return max_emg


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")