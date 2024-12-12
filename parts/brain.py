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
			start_signal = 0
			for i in range(20):
				total_image = self.sensor.get()
				self.base_images.append(total_image)
			base_images = np.array(self.base_images)
			self.base_image = np.mean(base_images, axis=0)

		images = self.sensor.get()
		images = images - self.base_image
		images /= 1500
		visual_image = copy.deepcopy(images[-1]) * 255
		visual_image = np.clip(visual_image, 0, 255)
		visual_image = cv2.resize(visual_image.astype(np.uint8), (500, 500))

		cv2.imshow("Pressure", visual_image)
		if cv2.waitKey(1) & 0xff == 27:
			break
		# _, angle, speed = self.model(images, hmd_yaw=0)
		angle = 0
		speed = 0

		visual_image = images[-1]
		# print(visual_image)
		# if hasattr(self.model, "visualized_image"):
		# 	print('hi')
			visual_image = self.model.visualized_image
		# if not visualize(visual_image):
		# 	return

		main_fps = round(self.fps_monitor.getFps())
		sensor_fps = self.sensor.fps

		if self.logger:
			self.logger.info(f"[Brain] sensor FPS:{sensor_fps}, main FPS: {main_fps}, Angle:{angle}, Speed:{speed}")

		return angle, speed

	def terminate(self):
		self.sensor.close()
		if self.logger:
			self.logger.info(f'[Brain] terminate Brain')
