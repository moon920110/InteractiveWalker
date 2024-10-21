import torch
import copy
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def get_foot_infos(images_, denoise=0, blur=0):
    images = copy.deepcopy(images_)

    '''
    Not use for now
    if denoise > 0:
        for _ in range(denoise):
            image = drop_noise(image)

    if blur > 0:
        for _ in range(blur):
            image = blur_image(image)
    '''
    images = images ** 3
    foot_signal_lower_bound = images.mean() + 3.0 * images.std()
    foot_x = np.where(images >= foot_signal_lower_bound)[1].astype(int)
    foot_y = np.where(images >= foot_signal_lower_bound)[2].astype(int)

    if foot_x.size> 3:
        foot_cordination = list(zip(foot_x, foot_y))
        cluster = KMeans(n_clusters=2, random_state=0).fit(foot_cordination)
    else:
        cluster = None

    return cluster

class FootDetector:
    def __init__(self, visualize):
        self.visualize = visualize

    def __call__(self, images, **kwargs):
        '''
        INPUT
        images : (time, 64, 64)
        OUTPUT
        available: True if result angle and speed are available
        angle: angle of human
        speed: speed of human
        '''

        foot_infos = get_foot_infos(images)
        available = True
        if foot_infos != None:
            #produce angle
            centers_of_foot = foot_infos.cluster_centers_
            slope = ((64 - centers_of_foot[0][0]) - (64 - centers_of_foot[1][0])) / \
                    (centers_of_foot[0][1] - centers_of_foot[1][1])
            orthogonal_slope = -1 / slope
            angle_candidate1 = np.degrees(np.arctan(orthogonal_slope))
            angle_candidate1 = angle_candidate1 - 90
            angle_candidate1 = angle_candidate1 * (-1)
            angle_candidate2 = angle_candidate1 - 180

            if hmd_yaw >= 0:
                if abs(angle_candidate1 - hmd_yaw) <= 90:
                    angle = angle_candidate1
                else:
                    angle = angle_candidate2
            else:
                if abs(angle_candidate2 - hmd_yaw) <= 90:
                    angle = angle_candidate2
                else:
                    angle = angle_candidate1

            # produce speed
            foot_track_start_timing = -1
            previous_foot_position = -1
            both_foot_count = 0

            for i in range(len(images[:])):
                max_pressure_index = np.unravel_index(images[i].argmax(), images[i].shape)
                current_foot_position = foot_infos.predict([list(max_pressure_index)])
                if i == 0:
                    previous_foot_position = current_foot_position
                if previous_foot_position != current_foot_position:
                    if foot_track_start_timing == -1:
                        foot_track_start_timing = i
                        previous_foot_position = current_foot_position
                    else:
                        foot_interval = i - foot_track_start_timing
                        break

                foot_signal_lower_bound = images.mean() + 3.0 * images.std()
                pressure_x = np.where(images[-i] >= foot_signal_lower_bound)[0].astype(int)
                pressure_y = np.where(images[-i] >= foot_signal_lower_bound)[1].astype(int)
                pressure_indexs = list(zip(pressure_x, pressure_y))
                if pressure_indexs != []:
                    foot_position_list = foot_infos.predict(pressure_indexs)
                    foot_position = Counter(list(foot_position_list)).most_common(2)

                    if len(foot_position) == 2:
                        both_foot_count += 1

                if both_foot_count == 5:
                    foot_interval = 1000000
                    break

                # if i == 10:
                #     foot_interval = 1000000
                #     break

                curr_speed = (8 - foot_interval) * 150  ## (Frame per second/step per frame) * 60 second = steps per min
                if not foot_interval == 100000 and curr_speed >= 0:
                    # self.speed = 0.2 * self.speed + 0.8 * curr_speed * type
                    speed = curr_speed
                    if speed >= 750:
                        speed = 750
                else:
                    speed = 0

        else:
            available, angle, speed = False, None, None
        return available, angle, speed


