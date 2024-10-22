from tactile_collecting.sensors.sensors import SensorEnv
from tactile_collecting.model.VisionModel_isaac import FootDetector as isaac_model
from tactile_collecting.sensors.app.FramerateMonitor import FramerateMonitor
from utils.utils import visualize

class Brain:
	def __init__(self, logger=None):
		self.model = None
		self.fps_monitor = None
		self.sensor = None
		self.logger = logger

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
		images = self.sensor.get()
		# _, angle, speed = self.model(images, hmd_yaw=0)
		angle = 0
		speed = 0

		visual_image = images[-1]
		if hasattr(self.model, "visualized_image"):
			visual_image = self.model.visualized_image
		if not visualize(visual_image):
			return

		main_fps = round(self.fps_monitor.getFps())
		sensor_fps = self.sensor.fps

		if self.logger:
			self.logger.info(f"[Brain] sensor FPS:{sensor_fps}, main FPS: {main_fps}, Angle:{angle}, Speed:{speed}")

		return angle, speed

	def terminate(self):
		self.sensor.close()
		if self.logger:
			self.logger.info(f'[Brain] terminate Brain')
