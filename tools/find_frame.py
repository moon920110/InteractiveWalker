import os
import numpy as np
import h5py
import cv2

data_path = '../data/raw_data/'

exercise_list = ['/squat_video', '/squat_pose', '/squat_tactile', '/lunge_video', '/lunge_pose',
              '/lunge_tactile', '/crunch_video', '/crunch_pose', '/crunch_tactile']

person_list = [#'BCM', 'BIC', 'CYJ', 'HTG', 'JHY', MJY', 'JHC', 'JIT',
               # 'KHM', 'LSY', 'MMN', 'NEK',
                'NWJ', 'OSW', 'PSY2', 'PYM', 'YWS', 'JEH']

for person in person_list:
    for exercise in exercise_list:

        input_data = data_path + person + exercise
        for sub_dir in os.listdir(input_data):
            d = os.path.join(input_data, sub_dir)
            if d.endswith("left_frame.avi"):
                left_video = cv2.VideoCapture(d)
                frame_cnt = 0
                while True:
                    ret, frame = left_video.read()
                    if not ret:
                        break
                    print("{} do {}. {} frames of {}"
                          .format(person, exercise, frame_cnt, left_video.get(cv2.CAP_PROP_FRAME_COUNT)))
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(40) & 0xFF == ord('q'):
                        break
                    frame_cnt += 1
                left_video.release()
                cv2.destroyAllWindows()
