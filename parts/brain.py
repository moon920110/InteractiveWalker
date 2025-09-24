import copy
from tactile_collecting.sensors.sensors import SensorEnv
from tactile_collecting.model.VisionModel_isaac import FootDetector as isaac_model
from tactile_collecting.sensors.app.FramerateMonitor import FramerateMonitor
from utils.utils import visualize
import numpy as np
import cv2 as cv2


class Brain:
    def __init__(self, stage, logger=None):
        self.model = None
        self.fps_monitor = None
        self.sensor = None
        self.logger = logger
        self.base_images = []
        self.start_signal = 1
        self.base_image = None
        self.right_arm_cy = None
        self.right_arm_cx = None
        self.right_arm_x_range = None
        self.right_arm_y_range = None
        self.left_arm_cx = None
        self.left_arm_cy = None
        self.left_arm_x_range = None
        self.left_arm_y_range = None
        self.stage = stage
        self.bad_row_indexs = [16, 27]
        self.bad_col_indexs = [9]

    def init(self, ports=["/dev/ttyUSB0"], stage=None):
        try:
            self.model = isaac_model(visualize=True)
            self.fps_monitor = FramerateMonitor()

            if self.logger:
                self.logger.info("[Brain] initializing sensors...")
            self.sensor = SensorEnv(
                ports=ports,
                stack_num=20,
                adaptive_calibration=False,
                stage=stage,
                normalize=True
            )
            if self.logger:
                self.logger.info("[Brain] sensor init finish")
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"[Brain] sensor init error: {e}")
            return False

    def test_sensor(self):
        while True:
            images = self.sensor.get()
            if not visualize(images[-1]):
                break
            print(f"sensor FPS : {self.sensor.fps}")

    def think(self):
        if self.start_signal == 1:
            self.start_signal = 0
            for i in range(50):
                total_image = self.sensor.get()
                self.base_images.append(total_image[-1])
            self.stage.put('collect')
            base_images = np.array(self.base_images[:-30])

            self.base_image = np.max(base_images, axis=0)
            self.base_image = self.base_image - np.min(self.base_image)
            np.savetxt('base.csv', self.base_image, delimiter=',')


            temp = np.array(np.transpose(
                np.array([self.base_image[21:, :] / 100, self.base_image[21:, :] / 100, self.base_image[21:, :] / 100]),
                (1, 2, 0)), dtype='uint8')
            temp[temp <= 5] = 0
            temp = cv2.cvtColor(cv2.UMat(temp), cv2.COLOR_BGR2GRAY)
            right_arm_cnts, _ = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            sorted_right_arm_cnts = sorted(right_arm_cnts, key=cv2.contourArea, reverse=True)
            x, y, w, h = cv2.boundingRect(sorted_right_arm_cnts[0])
            print(x, y, w, h)
            M = cv2.moments(sorted_right_arm_cnts[0])
            self.right_arm_cx = int(M['m10'] / M['m00'])
            self.right_arm_cy = int(M['m01'] / M['m00'])
            self.right_arm_x_range = w
            self.right_arm_y_range = h

            temp = np.array(np.transpose(
                np.array([self.base_image[10:21, :] / 100, self.base_image[10:21, :] / 100,
                          self.base_image[10:21, :] / 100]),
                (1, 2, 0)), dtype='uint8')
            temp[temp <= 5] = 0
            temp = cv2.cvtColor(cv2.UMat(temp), cv2.COLOR_BGR2GRAY)
            left_arm_cnts, _ = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            sorted_left_arm_cnts = sorted(left_arm_cnts, key=cv2.contourArea, reverse=True)
            x, y, w, h = cv2.boundingRect(sorted_left_arm_cnts[0])
            print(x, y, w, h)
            M = cv2.moments(sorted_left_arm_cnts[0])
            self.left_arm_cx = int(M['m10'] / M['m00'])
            self.left_arm_cy = int(M['m01'] / M['m00'])
            self.left_arm_x_range = w
            self.left_arm_y_range = h

        images = self.sensor.get()
        # image = np.clip(images[-1] - self.base_image, 0, 1500)
        image = np.clip(images[-1], 0, 2500).astype(np.float64)
        image /= float(2500)
        x_from = self.right_arm_cx - int(self.right_arm_x_range / 2)
        x_end = self.right_arm_cx + int(self.right_arm_x_range / 2)
        # forback = np.mean(image[21:, :][:, self.right_arm_cx:self.right_arm_cx + int(self.right_arm_x_range * 0.7)]) -\
        # 		  np.mean(image[21:, :][:, self.right_arm_cx - int(self.right_arm_x_range * 0.7):self.right_arm_cx]) # + forward
        # forback = np.sum(image[21:, :][:, self.right_arm_cx:self.right_arm_cx + int(self.right_arm_x_range * 0.7)] > 0.3) +\
        # 		  np.sum(image[10:21, :][:, self.left_arm_cx:self.left_arm_cx + int(self.left_arm_x_range * 0.7)] > 0.3) - \
        # 		  np.sum(image[21:, :][:, self.right_arm_cx - int(self.right_arm_x_range * 0.7):self.right_arm_cx] > 0.3) - \
        # 		  np.sum(image[10:21, :][:, self.left_arm_cx - int(self.left_arm_x_range * 0.7):self.left_arm_cx] > 0.3) # + forward
        forback = np.mean(image[21:, :][:, self.right_arm_cx:self.right_arm_cx + int(self.right_arm_x_range * 0.6)]) + \
                  np.mean(image[10:21, :][:, self.left_arm_cx:self.left_arm_cx + int(self.left_arm_x_range * 0.6)]) - \
                  np.mean(image[21:, :][:, self.right_arm_cx - int(self.right_arm_x_range * 0.6):self.right_arm_cx]) - \
                  np.mean(image[10:21, :][
                              :, self.left_arm_cx - int(self.left_arm_x_range * 0.6):self.left_arm_cx])  # + forward
        leftright = np.mean(image[21:, :][:self.right_arm_cy, x_from:x_end]) + \
                    np.mean(image[10:21, :][self.left_arm_cy:, x_from:x_end]) - \
                    np.mean(image[21:, :][self.right_arm_cy:, x_from:x_end]) - \
                    np.mean(image[10:21, :][:self.left_arm_cy, x_from:x_end])  # + right
        print(np.mean(image[21:, :][:, :7]))
        if np.mean(image[21:, :][:, :7]) >= 0.90:
            STS = True
        else:
            STS = False

        # print(forback, leftright)
        visual_image = copy.deepcopy(image) * 255
        visual_image = np.clip(visual_image, 0, 255)
        visual_image = cv2.resize(visual_image.astype(np.uint8), (500, 500))

        cv2.imshow("Pressure", visual_image)
        # if cv2.waitKey(1) & 0xff == 27:
        #     break
        # _, angle, speed = self.model(images, hmd_yaw=0)
        # angle = 0
        # speed = 0

        # visual_image = images[-1]
        # print(visual_image)
        # if hasattr(self.model, "visualized_image"):
        # 	print('hi')
        # 	visual_image = self.model.visualized_image
        # if not visualize(visual_image):
        # 	return

        main_fps = round(self.fps_monitor.getFps())
        sensor_fps = self.sensor.fps

        # if self.logger:
        # 	self.logger.info(f"[Brain] sensor FPS:{sensor_fps}, main FPS: {main_fps}, Angle:{angle}, Speed:{speed}")

        return forback, leftright, STS

    def terminate(self):
        self.sensor.close()
        if self.logger:
            self.logger.info(f'[Brain] terminate Brain')
