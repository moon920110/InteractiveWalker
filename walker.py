import datetime
import logging
import threading
import serial
import time
import queue
import argparse
from parts.brain import Brain
import pyrealsense2 as rs
import numpy as np
import cv2


def detect_obstacles(depth_frame, depth_scale, threshold=1.5):
    """Detects obstacles closer than a given threshold (meters)"""
    depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale  # Convert depth to meters
    mask = (depth_image > 0) & (depth_image < threshold)  # Highlight obstacles closer than threshold
    return mask, depth_image
def detect_obstacle_left(depth_frame, depth_scale, threshold=1.5):
    """Detects obstacles closer than a given threshold (meters)"""
    depth_image = np.asanyarray(depth_frame.get_data())[200:280, 170:270] * depth_scale  # Convert depth to meters
    mask = (depth_image > 0) & (depth_image < threshold)  # Highlight obstacles closer than threshold
    return mask, depth_image
def detect_obstacle_mid(depth_frame, depth_scale, threshold=1.5):
    """Detects obstacles closer than a given threshold (meters)"""
    depth_image = np.asanyarray(depth_frame.get_data())[200:280, 270:370] * depth_scale  # Convert depth to meters
    mask = (depth_image > 0) & (depth_image < threshold)  # Highlight obstacles closer than threshold
    return mask, depth_image

def detect_obstacle_right(depth_frame, depth_scale, threshold=1.5):
    """Detects obstacles closer than a given threshold (meters)"""
    depth_image = np.asanyarray(depth_frame.get_data())[200:280, 370:470] * depth_scale  # Convert depth to meters
    mask = (depth_image > 0) & (depth_image < threshold)  # Highlight obstacles closer than threshold
    return mask, depth_image

def compute_distance(depth_image, mask):
    """Compute average distance of obstacles from the camera"""
    obstacle_depth_values = depth_image[mask]
    if len(obstacle_depth_values) > 0:
        average_distance = np.mean(obstacle_depth_values)
        return average_distance
    else:
        return None  # No obstacles

class Walker:
    def __init__(self):
        self.forback = 0
        self.leftright = 0
        self.STS = False
        self.isStand = True
        self.STS_flag = False
        self.key_flag = False
        self.leftright_flag = False
        self.keyInput = ''
        self.start_signal = 1
        self.left_turn_scale = 1
        self.right_turn_scale = 1

        # TODO: IMU
        self.tilt = -50

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        fh = logging.FileHandler(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.brain = Brain(logger=self.logger)

        self.stop_event = threading.Event()
        self.init()


    def init(self):
        brain_check = self.brain.init()



    def _run_imu(self, keyQueue, mode):
        arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)
        i = 0
        while not self.stop_event.is_set():

            # try:
            #     print(float(arduino.readline().decode('utf-8').strip()))
            # except: pass

            # if keyQueue.empty:
            #     pass
            # else:
            # keyInput = keyQueue.get()
            # if keyInput != '':
            #     self.key_flag = True
            # print('sts:', self.STS_flag, 'key: ', self.key_flag, 'leftright: ', self.leftright)
            if self.start_signal:
                self.start_signal = 0
                print(mode)
                print('mode set')
                arduino.write(mode.encode('utf-8'))
                time.sleep(0.1)
                self.temp = arduino.readline().decode('utf-8')
                print(self.temp)
            #     arduino.write('init'.encode('utf-8'))
            #     time.sleep(0.1)
            #     print('Init set')
            i += 1
            # print(i)
            if mode == 'full':
                if self.STS_flag:
                    if self.isStand:
                        command = 'down'
                    else:
                        command = 'up'
                    # print(command)
                    arduino.write(command.encode('utf-8'))
                    time.sleep(10)
                elif self.leftright_flag:
                    if self.leftright > 0:
                        command = 'S1 ' + str(int(500 * self.leftright)) + ',S2 ' + str(int(500 * self.leftright)) + ",D1 0,D2 1,"
                    else:
                        command = 'S1 ' + str(int(500 * (-self.leftright))) + ',S2 ' + str(int(500 * (-self.leftright))) + ",D1 1,D2 0,"
                    arduino.write(command.encode('utf-8'))
                    time.sleep(0.1)
                elif self.key_flag:
                    self.key_flag = False
                    command = self.keyInput
                    arduino.write(command.encode('utf-8'))
                    time.sleep(0.1)
                else:
                    command = 'S1 ' + str(int(13 * self.forback * self.left_turn_scale)) + ',S2 ' + str(int(13 * self.forback * self.right_turn_scale)) + ",D1 0,D2 0,"
                    # arduino.write('S1 0,S2 0,D1 0,D2 0'.encode('utf-8'))
                    arduino.write(command.encode('utf-8'))
                    # print(self.angle, self.speed, 'write')
                    time.sleep(0.1)
                    # self.temp = arduino.readline().decode('utf-8')
                    # print(self.temp)
                # print(command)
                self.temp = arduino.readline().decode('utf-8')
                print(self.temp)
        # pass

    def _run_brain(self):
        while not self.stop_event.is_set():
            self.forback, self.leftright, self.STS = self.brain.think()
            # print(self.forback)
            if self.leftright <= -0.15 or self.leftright >= 0.15:
                self.forback = 0
                self.leftright_flag = True
            else:
                self.leftright_flag = False
            if self.leftright >= -0.03 and self.leftright <= 0.03:
                self.leftright = 0
            if self.forback >= -0.03 and self.forback <= 0.03:
                self.forback = 0

            if self.forback != 0 and self.leftright != 0:
                self.STS = False
                self.isStand = True
            if self.STS == True and self.isStand == True:
                self.STS_flag = True
                time.sleep(10)
                self.STS_flag = False
                self.isStand = False
                continue
            if self.STS == True and self.isStand == False:
                self.STS_flag = True
                time.sleep(10)
                self.STS_flag = False
                self.isStand = True
                continue

    # def _run_keyinput(self, keyQueue):
    #     while not self.stop_event.is_set():
    #         key = input()
    #         self.key_flag = True
    #         self.keyInput = key
    #         print(key, 'put')
    #         # keyQueue.put(key)

    def _run_camera(self):
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)

        # Get RealSense intrinsics
        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        # Depth scaling factor
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        while not self.stop_event.is_set():
            time.sleep(0.1)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                continue

            # Detect obstacles within 1.5 meters
            obstacle_mask, depth_image = detect_obstacle_left(depth_frame, depth_scale, threshold=1.0)
            # Compute the average distance of obstacles
            left_average_distance = compute_distance(depth_image, obstacle_mask)
            obstacle_mask, depth_image = detect_obstacle_mid(depth_frame, depth_scale, threshold=1.0)
            mid_average_distance = compute_distance(depth_image, obstacle_mask)
            obstacle_mask, depth_image = detect_obstacle_right(depth_frame, depth_scale, threshold=1.0)
            right_average_distance = compute_distance(depth_image, obstacle_mask)
            obstacle_mask, depth_image = detect_obstacles(depth_frame, depth_scale, threshold=1.0)

            # Normalize depth image for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=50), cv2.COLORMAP_JET)

            # Mark obstacles within 3 meters in red
            depth_colormap[obstacle_mask] = [0, 0, 255]

            # Display depth map with obstacle marking
            cv2.imshow("Depth Map with Obstacles within 3m", depth_colormap)

            # Display the average distance of obstacles within 3 meters
            if left_average_distance is not None and mid_average_distance is not None:
                print(f"There is obstacle on your left. Avoiding to right")
                self.right_turn_scale = 0.5
            elif mid_average_distance is not None and right_average_distance is not None:
                print(f"There is obstacle on your right. Avoiding to left")
                self.left_turn_scale = 0.5
            elif right_average_distance is not None and left_average_distance is not None and mid_average_distance is not None:
                if left_average_distance <= right_average_distance:
                    print(f"There is obstacle on your left. Avoiding to right")
                    self.right_turn_scale = 0.5
                if left_average_distance >= right_average_distance:
                    print(f"There is obstacle on your right. Avoiding to left")
                    self.left_turn_scale = 0.5
                # print(f"Average distance of right obstacles within 1.5 meters: {right_average_distance:.2f} meters")
            else:
                self.left_turn_scale = 1
                self.right_turn_scale = 1
            # else:
            #     print("No obstacles within 1.5 meters detected.")

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def run_walker(self, args):
        keyQueue = queue.Queue()
        mode = args.mode
        imu_thread = threading.Thread(target=self._run_imu, args=(keyQueue, mode))
        # if mode == 'full':
        brain_thread = threading.Thread(target=self._run_brain)
        # camera_thread = threading.Thread(target=self._run_camera)
        # keyinput_thread = threading.Thread(target=self._run_keyinput, args=(keyQueue,))

        try:
            imu_thread.start()
            self.logger.info(f'[Walker] imu thread start')
            # keyinput_thread.start()
            # self.logger.info(f'[Walker] KeyInput thread start')
            # if mode == 'full':
            brain_thread.start()
            self.logger.info(f'[Walker] Brain thread start')
            # camera_thread.start()
            # self.logger.info(f'[Walker] camera thread start')
            brain_thread.join()
            # camera_thread.join()
            # keyinput_thread.join()
            imu_thread.join()


        except KeyboardInterrupt:
            self.logger.error("[Walker] KeyboardInterrupt")

        finally:
            self.stop_event.set()
            self._terminate()
            self.logger.info("[Walker] terminate Walker")

    def _terminate(self):
        self.brain.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--mode", type=str, default='full') # full, pressure
    args = parser.parse_args()
    # arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=10)
    # time.sleep(5)
    # arduino.write('init\n'.encode('utf-8'))
    # mode = args.mode + '\n'
    # arduino.write(mode.encode('utf-8'))
    # time.sleep(5)
    walker = Walker()
    walker.run_walker(args)