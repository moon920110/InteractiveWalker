import pickle
import os
import glob
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd


def update_frame(frame):
    """지정된 프레임을 화면에 표시합니다."""
    im.set_data(video[frame])
    ax.set_title(f"Frame {frame + 1} of {frame_len}")
    plt.draw()


def on_key(event):
    """키보드 이벤트 처리 함수."""
    global current_frame
    if event.key == 'right':
        current_frame = min(current_frame + 1, frame_len - 1)  # 다음 프레임으로 이동
    elif event.key == 'up':
        current_frame = min(current_frame + 10, frame_len - 1)  # 다음 프레임으로 이동
    elif event.key == 'left':
        current_frame = max(current_frame - 1, 0)  # 이전 프레임으로 이동
    elif event.key == 'down':
        current_frame = max(current_frame - 10, 0)  # 이전 프레임으로 이동
    update_frame(current_frame)

def update_frame_tac(frame):
    """지정된 프레임을 화면에 표시합니다."""
    im_tac.set_data(tactile[frame])
    ax_tac.set_title(f"Frame {frame + 1} of {frame_len}")
    plt.draw()


def on_key_tac(event):
    """키보드 이벤트 처리 함수."""
    global current_frame
    if event.key == 'right':
        current_frame = min(current_frame + 1, frame_len - 1)  # 다음 프레임으로 이동
    elif event.key == 'up':
        current_frame = min(current_frame + 10, frame_len - 1)  # 다음 프레임으로 이동
    elif event.key == 'left':
        current_frame = max(current_frame - 1, 0)  # 이전 프레임으로 이동
    elif event.key == 'down':
        current_frame = max(current_frame - 10, 0)  # 이전 프레임으로 이동
    update_frame_tac(current_frame)


def visualize_tactile(tactile):
    global im_tac, ax_tac
    fig, ax_tac = plt.subplots()
    im_tac = ax_tac.imshow(tactile[current_frame])
    ax_tac.axis('off')
    fig.canvas.mpl_connect('key_press_event', on_key_tac)  # 키보드 이벤트에 함수 연결
    plt.show()
    plt.gcf().canvas.mpl_connect('key_press_event', close)



def close(event):
    if event.key == 'q':
        plt.close()

def visualize_emg(emg, exercise):
    title_list = ['LT TIB.ANT', 'LT LAT. GASTRO', 'LT VLO',
            'LT SEMITEND', 'LT GLUT', 'LT RECT.ABDOM',
            'RT TIB.ANT', 'RT LAT. GASTRO', 'RT VLO',
            'RT SEMITEND', 'RT GLUT', 'RT RECT.ABDOM']

    if exercise =='squat':
        check_index = 2
        check2_index = 8
    if exercise == 'lunge':
        check_index = 2
        check2_index = 8
    if exercise =='crunch':
        check_index = 5
        check2_index = 11

    emg_frame_len = 500
    peaks, _ = find_peaks(-emg[0:emg_frame_len, check_index])
    plt.figure(figsize=(10, 6))
    plt.plot(emg[0:emg_frame_len, check2_index], label=title_list[check2_index])
    plt.plot(emg[0:emg_frame_len, check_index], label=title_list[check_index])
    plt.title(title_list[check_index])
    plt.legend()
    # plt.xticks(np.arange(0, emg_frame_len, 10))
    plt.xlabel(peaks)
    plt.show()
    plt.gcf().canvas.mpl_connect('key_press_event', close)
    print()


def visualize_video():
    global im, ax
    fig, ax = plt.subplots()
    im = ax.imshow(video[current_frame])
    ax.axis('off')
    # update_frame(current_frame)  # 초기 프레임 표시
    fig.canvas.mpl_connect('key_press_event', on_key)  # 키보드 이벤트에 함수 연결
    plt.show()
    plt.gcf().canvas.mpl_connect('key_press_event', close)

bad_data_list = ['BCM_crunch_tactile_origin', 'KHM_crunch_pose_origin', 'NWJ_crunch_video_origin', 'MJY_lunge_video_origin',
                 'MMN_lunge_tactile_origin', 'NEK_lunge_tactile_origin', 'NEK_lunge_video_origin', 'PSY_lunge1_origin', ]


# WHILE TIME MATCHING END
exercise = 'crunch'
# directory_path = '/Volumes/YH_SSD/(2024.03.25)_lunge_data'
# directory_path = '/Volumes/T7/(2024.03.20)jinha_data_for_sink'
directory_path = '/Volumes/T7/(2024.03.25)calibrated_data'

pattern = os.path.join(directory_path, f'*{exercise}*')
file_list = glob.glob(pattern)

file_list.sort()
print(len(file_list))
for i, file in enumerate(file_list):
    if file.split('/')[-1]+'_origin' in bad_data_list : continue
    # if i < 2:
    #     continue
    with open(file, 'rb') as f:
        current_frame = 0
        data = pickle.load(f)
        print(i, file)
        frame_len = 500
        tactile = data[1]
        visualize_tactile(data[1])
        visualize_emg(data[2], exercise)
        video = data[3]
        visualize_video()


# # AFTER TIME MATCHING END
# # data_path = '/Volumes/YH_SSD/(2024.03.21)_trainee_data_all/'
# data_path = '/Volumes/T7/(2024.03.25)calibrated_data_before/'
# save_path = '/Volumes/T7/(2024.03.25)calibrated_data/'
# data_list = pd.read_csv('/Volumes/T7/emg_calibration.csv')
#
# bad_data_list = ['BCM_crunch_tactile_origin', 'KHM_crunch_pose_origin', 'NWJ_crunch_video_origin', 'MJY_lunge_video_origin',
#                  'MMN_lunge_tactile_origin', 'NEK_lunge_tactile_origin', 'NEK_lunge_video_origin', 'PSY_lunge1_origin', ]
#
# for filename in os.listdir(data_path):
#     if filename not in bad_data_list: continue
#     if filename in list(data_list['file_name']):
#         diff = int(data_list[data_list['file_name'] == filename]['diff'])
#         real_diff = diff + 15
#         if 'squat' in filename or 'crunch' in filename:
#             real_diff = real_diff + 5
#         # if 'lunge' not in filename: continue
#         with open(data_path + filename, 'rb') as f:
#             data = pickle.load(f)
#
#         if not len(data[0]) == len(data[1])== len(data[2])==len(data[3])==len(data[4]):
#             print(filename, len(data[0])-len(data[0]), len(data[1])-len(data[0]), len(data[2])-len(data[0]),len(data[3])-len(data[0]), len(data[4])-len(data[0]))
#         if real_diff >= 0:
#             data[2] = data[2][real_diff:]
#             data[0] = data[0][:len(data[2])]
#             data[1] = data[1][:len(data[2])]
#             data[3] = data[3][real_diff - diff:len(data[2])+real_diff - diff]
#             data[4] = data[4][real_diff - diff:len(data[2])+real_diff - diff]
#         else:
#             data[0] = data[0][-real_diff:]
#             data[1] = data[1][-real_diff:]
#             data[3] = data[3][-diff:]
#             data[4] = data[4][-diff:]
#             data[2] = data[2][:len(data[1])]
#             data[3] = data[3][:len(data[1])]
#             data[4] = data[4][:len(data[1])]
#
#         print(filename, len(data[0])-len(data[0]), len(data[1])-len(data[0]), len(data[2])-len(data[0]),len(data[3])-len(data[0]), len(data[4])-len(data[0]))
#
#         with open(save_path + filename[:-7], 'wb') as f:
#             pickle.dump(data, f)
#
#         print('check')