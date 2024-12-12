import copy

from tactile_collecting.sensors.sensors import SensorEnv
from tactile_collecting.model.VisionModel_isaac import FootDetector as isaac_model
from tactile_collecting.sensors.app.FramerateMonitor import FramerateMonitor
from utils.utils import visualize
import numpy as np
import cv2 as cv2

class Brain:
	def __init__(self, logger=None):
		self.model = None
		self.fps_monitor = None
		self.sensor = None
		self.logger = logger
		self.base_images = []
		self.start_signal = 1
		self.base_image = None
		self.right_arm_cy = None
		self.right_arm_cx = None

	def init(self, ports=["/dev/ttyUSB0"]):
		try:
			self.model = isaac_model(visualize=True)
			self.fps_monitor = FramerateMonitor()

			if self.logger:
				self.logger.info("[Brain] initializing sensors...")
			self.sensor = SensorEnv(
				ports=ports,
				stack_num=20,
				adaptive_calibration=True,
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
			base_images = np.array(self.base_images[:-30])
			self.base_image = np.mean(base_images, axis=0)
			# temp = (self.base_image[21:,:]/100).astype(np.uint8).reshape(1, 11, 32)
			# temp = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
			temp = np.array(np.array([self.base_image[21:,:]/100,self.base_image[21:,:]/100,self.base_image[21:,:]/100]).reshape(11, 32, 3), dtype='uint8')
			print(np.mean(temp))
			temp2 = cv2.cvtColor(cv2.UMat(temp), cv2.COLOR_BGR2GRAY)
			right_arm_cnts, _ = cv2.findContours(temp2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			sorted_right_arm_cnts = sorted(right_arm_cnts, key=cv2.contourArea, reverse=True)
			M = cv2.moments(sorted_right_arm_cnts[0])
			self.right_arm_cx = int(M['m10'] / M['m00'])
			self.right_arm_cy = int(M['m01'] / M['m00'])

		print(self.right_arm_cy, self.right_arm_cx)
		# print(self.start_signal)
		images = self.sensor.get()
		images = images - self.base_image
		images /= 1500
		visual_image = copy.deepcopy(images[-1]) * 255
		visual_image = np.clip(visual_image, 0, 255)
		visual_image = cv2.resize(visual_image.astype(np.uint8), (500, 500))

		cv2.imshow("Pressure", visual_image)
		# if cv2.waitKey(1) & 0xff == 27:
		# 	break
		# _, angle, speed = self.model(images, hmd_yaw=0)
		angle = 0
		speed = 0

		visual_image = images[-1]
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

		return angle, speed

	def terminate(self):
		self.sensor.close()
		if self.logger:
			self.logger.info(f'[Brain] terminate Brain')
