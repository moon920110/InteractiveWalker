####################################################
# Make sure display is available while running code#
####################################################

import argparse
from preprocessing_utils import *
from heatmap_from_keypoint3D import heatmap_gen
import matplotlib.pyplot as plt
import copy
from PIL import Image, ImageDraw
import pickle
import cv2
import pandas as pd
import os
import time

parser = argparse.ArgumentParser(description='Calibrate video')
parser.add_argument('--user', type=str, default='AMS', help='USER name')
parser.add_argument('--istrainer', type=bool, default=False, help='when clap on right camera')
parser.add_argument('--data_path', type=str, default='../data/', help='Experiment path')
# parser.add_argument('--save_path', type=str, default='../data/training_data/', help='save path')
parser.add_argument('--save_path', type=str, default='/Volumes/YH_SSD/(2024.03.21)_trainee_data_all/', help='save path')
parser.add_argument('--start_exercise', type=int, default=1, help='when clap on right camera')
parser.add_argument('--calibrate', type=bool, default=False, help='Set true if you calibrate')
parser.add_argument('--trim_video', type=bool, default=False, help='Set true if you trim your video before run openpose')
parser.add_argument('--read_keypoints', type=bool, default=False, help='Set true if you read 2D json keypoints from OpenPose')
parser.add_argument('--trainable', type=bool, default=False, help='Set true if you generate trainable file structure')
parser.add_argument('--trainable_llm', type=bool, default=False, help='Set true if you generate trainable file structure')
parser.add_argument('--trainable_emg', type=bool, default=True, help='Set true if you generate trainable file structure')

args = parser.parse_args()

# tall_dict = {'LSE':158, 'PSY':167, 'AMS':179, 'LSJ':173, 'PJH':166, 'PSJ':171, 'HMH':188, 'JYG':174}
# weight_dict = {'LSE':50, 'PSY':51, 'AMS':75, 'LSJ':75, 'PJH':63, 'PSJ':70, 'HMH':112, 'JYG':79}

# skip_list = ['LSE', 'PSY', 'AMS', 'LSJ', 'PJH', 'PSJ', 'HMH', 'JYG', 'BCM', 'BIC', 'CYJ', 'HTG', 'JHY', 'MJY', 'JHC',
#              'JIT', 'KHM', 'LSY', 'MMN', 'NEK', 'NWJ', 'OSW', 'PSY2', ]

# skip_list = [] # finished 02 24

skip_list = ['LSE', 'PSY', 'AMS', 'LSJ', 'PJH', 'PSJ', 'HMH', 'JYG', 'BCM', 'BIC', 'CYJ', 'HTG', 'JHY', 'MJY', 'JHC',
             'JIT', 'KHM', 'LSY', 'MMN', 'NEK', 'NWJ']

# skip_list = ['LSE', 'PSY', 'AMS', 'LSJ', 'PJH', 'PSJ', 'HMH', 'JYG']
# skip_list = ['LSE', 'PSY', 'AMS', 'LSJ', 'PJH', 'PSJ', 'HMH', 'JYG', 'JIT']
# view_list = ['JYG_crunch1', 'PJH_squat2', 'PJH_squat1', 'LSJ_crunch2', 'LSJ_crunch1', 'LSJ_squat1', 'LSJ_squat2' ]


fault_data_list = ['PYM_crunch_video', # not working with preprocessing code
                   'CYJ_lunge_tactile', #(LT glut out of signal)
                   'CYJ_lunge_video', #(LT glut out of signal)
                   'NWJ_squat_video',]# all data recorded 1000
# fault_data_list = ['AMS_squat1', # i don't know why
#                    'AMS_lunge2',
#                    'PSY_crunch2',
#                    'PSJ_lunge1',
#                    'BCM_squat_pose',
#                    'BCM_squat_tactile'
#                    'BIC_squat_video',
#                    'OSW_squat_video',
#                    'PYM_crunch_video',
#                    'PSY2_crunch_pose',
#
#                    'CYJ_lunge_tactile', #(LT glut out of signal)
#                    'CYJ_lunge_video', #(LT glut out of signal)
#                    'NWJ_squat_video', # all data recorded 1000
#
#                    ]

balance_fix_data_dict = {
    'AMS_lunge1' : ['GLUT. MAX. (uV)'], 'AMS_squat2' : ['GLUT. MAX. (uV)'],
    'CYJ_squat_pose' : ['GLUT. MAX. (uV)'],
    'JEH_squat_pose' : ['GLUT. MAX. (uV)', 'SEMITEND. (uV)',  'VLO (uV)'],
    'JEH_squat_tactile' : ['GLUT. MAX. (uV)', 'SEMITEND. (uV)',  'VLO (uV)'],
    'JEH_squat_video' : ['GLUT. MAX. (uV)', 'SEMITEND. (uV)',  'VLO (uV)'],
    'JEH_lunge_pose' : ['GLUT. MAX. (uV)', 'SEMITEND. (uV)',  'VLO (uV)'],
    'JEH_lunge_tactile' : ['GLUT. MAX. (uV)', 'SEMITEND. (uV)',  'VLO (uV)'],
    'JEH_lunge_video' : ['GLUT. MAX. (uV)', 'SEMITEND. (uV)',  'VLO (uV)'],
    'JHC_crunch_pose' : ['RECT.ABDOM.UP. (uV)'],
    'NEK_crunch_pose' : ['RECT.ABDOM.UP. (uV)'],
    'NWJ_crunch_pose': ['RECT.ABDOM.UP. (uV)'],
    'NWJ_crunch_tactile': ['RECT.ABDOM.UP. (uV)'],
    'NWJ_crunch_video': ['RECT.ABDOM.UP. (uV)'],
    'OSW_crunch_video': ['RECT.ABDOM.UP. (uV)'],
    'OSW_crunch_pose': ['RECT.ABDOM.UP. (uV)'],
    'OSW_lunge_pose': ['VLO (uV)'], 'OSW_lunge_tactile': ['VLO (uV)'], 'OSW_lunge_video': ['VLO (uV)'],
    'OSW_squat_pose': ['VLO (uV)'], 'OSW_squat_tactile': ['VLO (uV)'], 'OSW_squat_video': ['VLO (uV)'],
}

replace_fix_data_dict = {
    'LSJ_squat1': 'SEMITEND. (uV)', 'LSJ_squat2': 'SEMITEND. (uV)',
    'NEK_squat_pose': 'RECT.ABDOM.UP. (uV)',
    'OSW_crunch_tactile': 'RECT.ABDOM.UP. (uV)',
    'PJH_squat1': 'LAT. GASTRO (uV)', 'PJH_squat2': 'LAT. GASTRO (uV)',
    'PSY2_squat_tactile': 'LAT. GASTRO (uV)',
    'YWS_crunch_tactile': 'GLUT. MAX. (uV)',
    'PSY_crunch1': 'GLUT. MAX. (uV)',
    'LSJ_crunch1': 'GLUT. MAX. (uV)', 'LSJ_crunch2': 'GLUT. MAX. (uV)',
    'JYG_crunch1': 'GLUT. MAX. (uV)'
}

new_fault_data_list = []

def data_fix(df_emg_list, name, exercise):
    if name + '_' + exercise in balance_fix_data_dict.keys():
        for i in range(len(balance_fix_data_dict[name+ '_' + exercise])):
            target = balance_fix_data_dict[name+ '_' + exercise][i]
            print(name + '_' + exercise + 'data' + target, 'fixed')
            high = max(df_emg_list['RT ' + target].mean(), df_emg_list['LT ' + target].mean())
            df_emg_list['RT ' + target] *= (high/df_emg_list['RT ' + target].mean())
            df_emg_list['LT ' + target] *= (high/df_emg_list['LT ' + target].mean())
            df_emg_list['RT ' + target] = df_emg_list['RT ' + target].clip(upper=1)
            df_emg_list['LT ' + target] = df_emg_list['LT ' + target].clip(upper=1)

    if name + '_' + exercise in replace_fix_data_dict.keys():
        target = replace_fix_data_dict[name + '_' + exercise]
        print(name + '_' + exercise + 'data' + target, 'replaced')
        if name in ['LSJ', 'NEK', 'OSW', 'PJH', 'PSY2', 'YWS', 'LSJ', 'JYG']:
            df_emg_list['RT ' + target] = df_emg_list['LT ' + target]
        if name in ['PSY']:
            df_emg_list['LT ' + target] = df_emg_list['RT ' + target]

    if name + '_' + exercise == 'JYG_crunch2':
        df_emg_list['RT GLUT. MAX. (uV)'] = 0
        df_emg_list['LT GLUT. MAX. (uV)'] = 0

    return df_emg_list


if args.calibrate:
    calibration_path = '../calibration_data/'
    for video in os.listdir(calibration_path):
        try:
            print(video)
            cam_cap(calibration_path + video)
            # vertices_checker(calibration_path + video)
        except NotADirectoryError:
            pass
    print('\nNow run calibrate.m and StereoTriangulate_carpet.m on MATLAB')


if args.trim_video:
    left_video_timestamp, right_video_timestamp = timestamp_matcher(hdf5)
    left_video = args.path + '/' + \
                 [file for file in os.listdir(args.path) if file.endswith('camera-left_frame.avi')][0]
    right_video = args.path + '/' + \
                  [file for file in os.listdir(args.path) if file.endswith('camera-right_frame.avi')][0]
    trim_video(left_video, right_video, left_video_timestamp, right_video_timestamp, args.path, hdf5)
    print('\nNow run OpenPose on Docker \nThen, run StereoTriangulate.m, m_to_csv.m on MATLAB')

if args.read_keypoints:
    json_to_csv(args.path)
    print('\nNow run StereoTriangulate.m and m_to_csv.m on MATLAB')

if args.trainable:
    to_keypoints3D(args.path)
    keypoints = args.path + '/keypoints/keypoints3D.csv'
    left_timestamp, right_timestamp = timestamp_matcher(hdf5, keypoints)
    left_closest_cam_idx, right_closest_cam_idx = closest_camera_matcher(hdf5, left_timestamp, right_timestamp)

    left_key_timestamp_idx, upper_left_upsampled_tactile = up_sampling_tactile(hdf5, 'left', left_closest_cam_idx, left_timestamp)
    right_key_timestamp_idx, upper_right_upsampled_tactile = up_sampling_tactile(hdf5, 'right', right_closest_cam_idx, right_timestamp)

    _, lower_left_upsampled_tactile = up_sampling_tactile(hdf5, 'c', left_closest_cam_idx, left_timestamp)
    _, lower_right_upsampled_tactile = up_sampling_tactile(hdf5, 'd', right_closest_cam_idx, right_timestamp)

    upper_left_upsampled_tactile = normalize_directly(upper_left_upsampled_tactile)
    upper_right_upsampled_tactile = normalize_directly(upper_right_upsampled_tactile)
    lower_left_upsampled_tactile = normalize_directly(lower_left_upsampled_tactile)
    lower_right_upsampled_tactile = normalize_directly(lower_right_upsampled_tactile)

    # upper_tactile = concat_tactiles(upper_left_upsampled_tactile, upper_right_upsampled_tactile)
    # lower_tactile = concat_tactiles(lower_left_upsampled_tactile, lower_right_upsampled_tactile)

    upsampled_tactile = concat_four_tactiles(upper_left_upsampled_tactile,
                                             upper_right_upsampled_tactile,
                                             lower_left_upsampled_tactile,
                                             lower_right_upsampled_tactile)

    print('Shape before padding :', upsampled_tactile.shape)
    padded_tactile = np.pad(upsampled_tactile, pad_width=((0, 0), (0, 32), (0, 32)))
    print('Shape after padding :', padded_tactile.shape)
    pickle.dump(padded_tactile, open(args.path + '/' + 'preprocessed_tactile.p', 'wb'))
    fitted_keypoints_to_p(keypoints, left_key_timestamp_idx, args.path)
    heatmap_gen(args.path)
    tmp_p_generator(args.path, 'preprocessed_tactile.p', 'fitted_keypoints.p_heatmap3D.p', 'fitted_keypoints.p_coord.p')
    # # trainable_p_generator(args.path, 'preprocessed_tactile.p', 'fitted_keypoints.p_heatmap3D.p', 'fitted_keypoints.p_coord.p')
    # # log_p(args.path)
    # print('\nPreprocessing is done!')

if args.trainable_emg:
    # load data list - organized by excel
    personal_info = pd.read_excel(args.data_path + 'data_personal_info.xlsx')
    frame_info = pd.read_excel(args.data_path + 'data_frame_info.xlsx')

    # TODO: use FOR sentence to bring each data
    #   for name in personal info list
    #       for exercise in data[data[name]=name]exercise
    #           bring tactile/emg data
    for name in personal_info['name']:
        if name in skip_list: continue
        # if name not in view_list: continue
        max_emg = make_max_emg(args.data_path, name, args.istrainer)
        tactile_base = make_base_tactile(args.data_path, name, args.istrainer)
        print(name, max_emg)

         #for test

        for exercise in frame_info[frame_info['name'] == name]['exercise']:
            begin = time.time()
            if name + '_' + exercise in fault_data_list:
                continue
            # if name + '_' + exercise not in view_list: continue

            try:
                # if name + '_' + exercise != 'HMH_crunch2':
                #     continue
                hdf5_path = args.data_path + 'raw_data/' + name + '/' + exercise + '/'
                hdf5 = hdf5_path + [file for file in os.listdir(hdf5_path) if file.endswith('hdf5')][0]
                # TODO: load data
                # set time range
                start_frame = frame_info[(frame_info['name']==name) & (frame_info['exercise']==exercise)]['frame_start'].values
                start_time = hdf5_to_np(hdf5, 'camera-right/frame_timestamp/time_s')[int(start_frame)]
                if args.istrainer:
                    end_time = start_time + 88
                else:
                    end_time = start_time + 58

                left_timestamp, right_timestamp, left_video, pose = timestamp_matcher(hdf5, hdf5_path)
                right_timestamp = [x for x in right_timestamp if x >= start_time]
                len_diff = len(left_timestamp) - len(right_timestamp)
                left_timestamp = left_timestamp[len_diff:]
                left_video = left_video[len_diff:]
                pose = pose[len_diff:]
                right_timestamp = [x for x in right_timestamp if x <= end_time]
                left_timestamp = left_timestamp[:len(right_timestamp)]
                left_video = left_video[:len(right_timestamp)]
                pose = pose[:len(right_timestamp)]

                # load tactile data
                left_closest_cam_idx, right_closest_cam_idx, c_closest_cam_idx, d_closest_cam_idx = closest_camera_matcher(
                    hdf5, left_timestamp, right_timestamp, start_time, end_time)

                _, upper_left_upsampled_tactile = up_sampling_tactile(hdf5, 'left', left_closest_cam_idx, right_timestamp, start_time, end_time)
                _, upper_right_upsampled_tactile = up_sampling_tactile(hdf5, 'right', right_closest_cam_idx, right_timestamp, start_time, end_time)
                _, lower_left_upsampled_tactile = up_sampling_tactile(hdf5, 'c', c_closest_cam_idx, right_timestamp, start_time, end_time)
                _, lower_right_upsampled_tactile = up_sampling_tactile(hdf5, 'd', d_closest_cam_idx, right_timestamp, start_time, end_time)

                temp_tactile = concat_four_tactiles(upper_left_upsampled_tactile,
                                                    upper_right_upsampled_tactile,
                                                    lower_left_upsampled_tactile,
                                                    lower_right_upsampled_tactile)

                # load emg data
                # # if user
                # column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
                #                'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT RECT.ABDOM.UP. (uV)', 'RT TIB.ANT. (uV)',
                #                'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)',
                #                'RT RECT.ABDOM.UP. (uV)', 'Switch (On)']
                # if trainer
                if args.istrainer:
                    column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
                               'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT LUMBAR ES (uV)', 'LT RECT.ABDOM.UP. (uV)',
                               'RT TIB.ANT. (uV)',
                               'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)',
                               'RT LUMBAR ES (uV)',
                               'RT RECT.ABDOM.UP. (uV)', 'Switch (On)']
                    preserve_column_name = [x for x in column_name if x != 'Activity' and x != 'Marker'
                                            and x != 'LT LUMBAR ES (uV)' and x != 'RT LUMBAR ES (uV)']
                else:
                    column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
                                   'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT RECT.ABDOM.UP. (uV)', 'RT TIB.ANT. (uV)',
                                   'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)',
                                   'RT RECT.ABDOM.UP. (uV)', 'Switch (On)']
                    preserve_column_name = [x for x in column_name if x != 'Activity' and x != 'Marker']
                emg_path = args.data_path + 'raw_data/' + name + '/EMG/' + exercise + '.csv'
                if len(pd.read_csv(emg_path, skiprows=4, low_memory=False).columns) == 17:
                    column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)',
                                   'LT VLO (uV)',
                                   'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT RECT.ABDOM.UP. (uV)',
                                   'RT TIB.ANT. (uV)',
                                   'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)',
                                   'RT RECT.ABDOM.UP. (uV)', 'Switch (On)', 'Switch (On)2']
                if len(pd.read_csv(emg_path, skiprows=4, low_memory=False).columns) == 17:
                    del df['Switch (On)2']
                df = pd.read_csv(emg_path, skiprows=4, names=column_name, low_memory=False)
                if args.istrainer:
                    df = df.drop(['Activity', 'Marker', 'LT LUMBAR ES (uV)', 'RT LUMBAR ES (uV)'], axis=1)
                else:
                    df = df.drop(['Activity', 'Marker'], axis=1)
                # TODO make exception for two markers
                condition = (df['Switch (On)'] != 0)  # 1 for trainer, 1000 for user
                filtered_df = df[condition].copy()
                filtered_df.reset_index(drop=True, inplace=True)
                # filtered_df = filtered_df.drop(['Activity', 'Marker'], axis=1)
                filtered_df['time'] = filtered_df['time'] - filtered_df.loc[0, 'time'] + start_time
                emg_list = []
                tmp = 0
                emg_timegap = filtered_df['time'][2] - filtered_df['time'][1]
                for i in range(len(right_timestamp)):
                    for j in range(tmp, len(filtered_df['time'])):
                        gap = abs(filtered_df['time'][j] - right_timestamp[i])
                        if gap < emg_timegap:
                            tmp = j + 10
                            emg_list.append(filtered_df.iloc[j])
                            break

                # normalize pressure
                temp_tactile = np.clip(temp_tactile - tactile_base, 0, 3000)
                high_pressure = np.quantile(temp_tactile[np.nonzero(temp_tactile)], 0.995)
                presure_norm_rate = 1000 / high_pressure
                temp_tactile = temp_tactile * presure_norm_rate
                temp_tactile = np.clip(temp_tactile, 0, 1000)

                # normalize size
                standard_tall = 180
                user_tall = int(personal_info[personal_info['name'] == name]['height'])
                change_pixel = int((standard_tall - user_tall) / standard_tall * 64)


                if exercise[:5] == 'lunge':
                    for idx in range(len(temp_tactile)):
                        temp = cv2.resize(temp_tactile[idx], (change_pixel + 64, change_pixel + 64),
                                          interpolation=cv2.INTER_CUBIC)
                        if change_pixel > 0:
                            temp = temp[:64, change_pixel:]
                        elif change_pixel < 0:
                            change = -1 * change_pixel
                            temp_zeros = np.zeros((64, 64))
                            temp_zeros[:64 - change, change:] = temp
                            temp = temp_zeros
                        temp = temp.reshape(1, 64, 64)
                        if idx == 0:
                            temp_array = temp
                        else:
                            temp_array = np.vstack((temp_array, temp))
                    temp_tactile = temp_array

                # TODO: crunch, squat에 대한 size-normalize,
                if exercise[:5] == 'crunc':
                    for idx in range(len(temp_tactile)):
                        temp = cv2.resize(temp_tactile[idx], (change_pixel + 64, change_pixel + 64),
                                          interpolation=cv2.INTER_CUBIC)
                        if change_pixel > 0:
                            temp = temp[change_pixel:, :64]
                        elif change_pixel < 0:
                            change = -1 * change_pixel
                            temp_zeros = np.zeros((64, 64))
                            temp_zeros[change:, :64 - change] = temp
                            temp = temp_zeros
                        temp = temp.reshape(1, 64, 64)
                        if idx == 0:
                            temp_array = temp
                        else:
                            temp_array = np.vstack((temp_array, temp))
                    temp_tactile = temp_array

                if exercise[:5] == 'squat':
                    change_half = int(change_pixel/2)
                    change_quater = int(change_pixel/4)
                    for idx in range(len(temp_tactile)):
                        temp = cv2.resize(temp_tactile[idx], (change_pixel + 64, change_pixel + 64),
                                          interpolation=cv2.INTER_CUBIC)
                        if change_pixel > 0:
                            temp = temp[(change_pixel-change_quater):(change_pixel-change_quater)+64, change_half:64+change_half]
                        elif change_pixel < 0:
                            change_pixel = -1 * change_pixel
                            change_half = -1 * change_half
                            change_quater = -1 * change_quater
                            temp_zeros = np.zeros((64, 64))
                            temp_zeros[(change_pixel-change_quater):(change_pixel-change_quater)+64, change_half:64-change_half] = temp
                            temp = temp_zeros
                        temp = temp.reshape(1, 64, 64)
                        if idx == 0:
                            temp_array = temp
                        else:
                            temp_array = np.vstack((temp_array, temp))
                    temp_tactile = temp_array


                # normalize emg
                df_emg_list = pd.DataFrame(emg_list)
                emg_norm_column_names = [x for x in preserve_column_name if x != 'time' and x != 'Switch (On)']
                for emg_norm_col_name in emg_norm_column_names:
                    df_emg_list[emg_norm_col_name] = df_emg_list[emg_norm_col_name] / max_emg[emg_norm_col_name]
                    df_emg_list[emg_norm_col_name] = df_emg_list[emg_norm_col_name].clip(upper=1)
                df_emg_list = data_fix(df_emg_list, name, exercise)

                # save all as images to check it works well or not
                temp_tactile_img = temp_tactile / 1000 * 255

                createDirectory('../test/' + name + '/' + exercise + '/')


                if exercise[:6] == 'crunch':
                    test_emg = list(df_emg_list['LT RECT.ABDOM.UP. (uV)'])
                else:
                    test_emg = list(df_emg_list['LT VLO (uV)'])

                # for i in range(temp_tactile_img.shape[0]):
                #     im = Image.fromarray(temp_tactile_img[i, :, :])
                #     im = im.convert("RGB")
                #     im = im.resize((500, 500))
                #     draw = ImageDraw.Draw(im)
                #     draw.text((200, 50), str(test_emg[i]), fill="red")
                #     im.save('../test/' + name + '/' + exercise + '/' + str(i) + '.png')
                #
                # for column in df_emg_list.columns:
                #     plt.clf()
                #     plt.plot(df_emg_list['time'], df_emg_list[column])
                #     plt.xlabel('time')
                #     plt.ylabel('EMG')
                #     plt.savefig('../test/' + name + '/' + exercise + '/0EMG_' + column + '.png')
                # df_emg_list.to_csv('../test/' + name + '/' + exercise + '/0_0.csv', sep=',')

                emg_numpy = np.array(df_emg_list)[:, 1:-1]

                # TODO: Normalize: each carpet sensitivity?
                # TODO: pose load, normalization
                concat_data = [right_timestamp[30:-5], temp_tactile[30:-5], emg_numpy[30:-5], left_video[30:-5], pose[30:-5]]
                with open(args.save_path + name + '_' + exercise, 'wb') as f:
                    pickle.dump(concat_data, f)
                print('Done {} {} data processing in {} seconds'.format(name, exercise, int(time.time()-begin)))
            except:
                print('new fault data:', name + '_' + exercise)
                new_fault_data_list.append(name + '_' + exercise)


    print('trainable_EMG')


if args.trainable_llm:
    # to_keypoints3D(args.path)
    # keypoints = args.path + '/keypoints/keypoints3D.csv'
    start_time = hdf5_to_np(hdf5, 'camera-right/frame_timestamp/time_s')[args.start_exercise]
    end_time = start_time + 90

    left_timestamp, right_timestamp = timestamp_matcher(hdf5)
    # left_closest_cam_idx, right_closest_cam_idx = closest_camera_matcher(hdf5, left_timestamp, right_timestamp)
    left_closest_cam_idx, right_closest_cam_idx, c_closest_cam_idx, d_closest_cam_idx = closest_camera_matcher(hdf5, left_timestamp, right_timestamp)

    left_key_timestamp_idx, upper_left_upsampled_tactile = up_sampling_tactile(hdf5, 'left', left_closest_cam_idx, right_timestamp)
    right_key_timestamp_idx, upper_right_upsampled_tactile = up_sampling_tactile(hdf5, 'right', right_closest_cam_idx, right_timestamp)

    _, lower_left_upsampled_tactile = up_sampling_tactile(hdf5, 'c', c_closest_cam_idx, right_timestamp)
    _, lower_right_upsampled_tactile = up_sampling_tactile(hdf5, 'd', d_closest_cam_idx, right_timestamp)

    # concat images
    temp_tactile = concat_four_tactiles(upper_left_upsampled_tactile,
                                             upper_right_upsampled_tactile,
                                             lower_left_upsampled_tactile,
                                             lower_right_upsampled_tactile)

    # find abnormal data positions
    temp_tactile_abnormal_value = np.median(temp_tactile, axis=0) + 1 * temp_tactile.std()

    # mean not abnormal data pixel values (make base tactile data)
    temp = copy.deepcopy(temp_tactile)
    temp[temp > temp_tactile_abnormal_value] = np.nan
    temp_base = np.nanmean(temp, axis=0)

    # take away base tactile data from concatenated tactile data (min=0)
    temp_tactile = temp_tactile - temp_base
    temp_tactile[temp_tactile < 0] = 0

    # # remove not pressured points
    # temp_tactile_abnormal_value = np.median(temp_tactile) + 1 * temp_tactile.std()
    # temp_tactile[temp_tactile < temp_tactile_abnormal_value] = 0

    #normalize image size
    standard_tall = 180
    user_tall = int(personal_info[personal_info['name'] == name]['height'])
    change_pixel = int((standard_tall-user_tall) / standard_tall * 64)

    for idx in range(len(temp_tactile)):
        temp = cv2.resize(temp_tactile[idx], (change_pixel+64, change_pixel+64), interpolation=cv2.INTER_CUBIC)
        if change_pixel > 0:
            temp = temp[:64, change_pixel:]
        elif change_pixel < 0:
            change = -1 * change_pixel
            temp_zeros = np.zeros((64, 64))
            temp_zeros[:64-change, change:] = temp
            temp = temp_zeros
        temp = temp.reshape(1, 64, 64)
        if idx == 0:
            temp_array = temp
        else: temp_array = np.vstack((temp_array, temp))
    temp_tactile = temp_array

    # normalize pressure
    high_pressure = np.quantile(temp_tactile[np.nonzero(temp_tactile)], 0.995)
    presure_norm_rate = 1000 / high_pressure
    temp_tactile = temp_tactile * presure_norm_rate
    temp_tactile = np.clip(temp_tactile, 0, 1000)

    # save all as images to check it works well or not
    temp_tactile_img = temp_tactile / 1000 * 255

    for i in range(temp_tactile_img.shape[0]):
        im = Image.fromarray(temp_tactile_img[i, :, :])
        im = im.convert("RGB")
        im = im.resize((500, 500))
        im.save('../test/' + str(i)+'.png')

    # test emg data draw graph
    # ## if user
    # column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
    #                'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT RECT.ABDOM.UP. (uV)', 'RT TIB.ANT. (uV)',
    #                'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)',
    #                'RT RECT.ABDOM.UP. (uV)', 'Switch (On)']
    ## if trainer
    column_name = ['time', 'Activity', 'Marker', 'LT TIB.ANT. (uV)', 'LT LAT. GASTRO (uV)', 'LT VLO (uV)',
                   'LT SEMITEND. (uV)', 'LT GLUT. MAX. (uV)', 'LT LUMBAR ES (uV)', 'LT RECT.ABDOM.UP. (uV)', 'RT TIB.ANT. (uV)',
                   'RT LAT. GASTRO (uV)', 'RT VLO (uV)', 'RT SEMITEND. (uV)', 'RT GLUT. MAX. (uV)', 'RT LUMBAR ES (uV)',
                   'RT RECT.ABDOM.UP. (uV)', 'Switch (On)']
    preserve_column_name = [x for x in column_name if x != 'Activity' and x != 'Marker']
    df = pd.read_csv(args.emg_path,  skiprows=4, names=column_name)
    df.drop(['Activity', 'Marker'], axis=1)
    condition = (df['Switch (On)'] == 1) # 1 for trainer, 1000 for user
    filtered_df = df[condition]
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df = filtered_df.drop(['Activity', 'Marker'], axis=1)

    filtered_df['time'] = filtered_df['time'] - filtered_df['time'].iloc[0] + start_time

    # checked! it smooth!
    plt.plot(filtered_df['time'].iloc[1:3000], filtered_df['LT TIB.ANT. (uV)'].iloc[1:3000])
    plt.xlabel('time')
    plt.ylabel('EMG')
    plt.show()

    # TODO: if data is ok, fit timeline
    # TODO: if data is smooth, del other timelines. if not, set average value as that timeline.
    right_timestamp = [x for x in right_timestamp if x >= start_time]
    temp_tactile_img = temp_tactile_img[len(temp_tactile_img) - len(right_timestamp):, :, :]
    right_timestamp = [x for x in right_timestamp if x <= end_time]
    temp_tactile_img = temp_tactile_img[:len(right_timestamp), :, :]

    emg_list = []
    tmp = 0
    emg_timegap = filtered_df['time'][2] - filtered_df['time'][1]
    for i in range(len(right_timestamp)):
        for j in range(tmp, len(filtered_df['time'])):
            gap = abs(filtered_df['time'][j] - right_timestamp[i])
            if gap < emg_timegap:
                tmp = j + 10
                emg_list.append(filtered_df.iloc[j])
                break
    concat_data = [right_timestamp, temp_tactile_img, emg_list]

    print('test')
    with open(args.save_path + name + '_' + exercise, 'wb') as f:
        pickle.dump(concat_data, f)


    # upper_left_upsampled_tactile = normalize_directly(upper_left_upsampled_tactile)
    # upper_right_upsampled_tactile = normalize_directly(upper_right_upsampled_tactile)
    # lower_left_upsampled_tactile = normalize_directly(lower_left_upsampled_tactile)
    # lower_right_upsampled_tactile = normalize_directly(lower_right_upsampled_tactile)

    # upper_tactile = concat_tactiles(upper_left_upsampled_tactile, upper_right_upsampled_tactile)
    # lower_tactile = concat_tactiles(lower_left_upsampled_tactile, lower_right_upsampled_tactile)

    # upsampled_tactile = concat_four_tactiles(upper_left_upsampled_tactile,
    #                                          upper_right_upsampled_tactile,
    #                                          lower_left_upsampled_tactile,
    #                                          lower_right_upsampled_tactile)



    # print('Shape before padding :', upsampled_tactile.shape)
    # padded_tactile = np.pad(upsampled_tactile, pad_width=((0, 0), (0, 32), (0, 32)))
    # print('Shape after padding :', padded_tactile.shape)
    # pickle.dump(padded_tactile, open(args.path + '/' + 'preprocessed_tactile.p', 'wb'))
    # fitted_keypoints_to_p(keypoints, left_key_timestamp_idx, args.path)
    # heatmap_gen(args.path)
    # tmp_p_generator(args.path, 'preprocessed_tactile.p', 'fitted_keypoints.p_heatmap3D.p', 'fitted_keypoints.p_coord.p')


    # # trainable_p_generator(args.path, 'preprocessed_tactile.p', 'fitted_keypoints.p_heatmap3D.p', 'fitted_keypoints.p_coord.p')
    # # log_p(args.path)
    # print('\nPreprocessing is done!')