import Jetson.GPIO as GPIO
import time


class StepMotorControl:
	def __init__(self, logger=True):
		self.logger = logger
		self.pul_pin_lf = 7
		self.dir_pin_lf = 11
		self.en_pin_lf = 13
		self.pul_pin_rf = 15
		self.dir_pin_rf = 16
		self.en_pin_rf = 18

	def init(self):
		# self.pul_pin_lf = pul_pin_lf
		# self.dir_pin_lf = dir_pin_lf
		# self.en_pin_lf = en_pin_lf
		# self.pul_pin_rf = pul_pin_rf
		# self.dir_pin_rf = dir_pin_rf
		# self.en_pin_rf = en_pin_rf

		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(self.pul_pin_lf, GPIO.OUT)
		GPIO.setup(self.dir_pin_lf, GPIO.OUT)
		GPIO.setup(self.en_pin_lf, GPIO.OUT)
		GPIO.setup(self.pul_pin_rf, GPIO.OUT)
		GPIO.setup(self.dir_pin_rf, GPIO.OUT)
		GPIO.setup(self.en_pin_rf, GPIO.OUT)

		GPIO.output(self.en_pin_lf, GPIO.HIGH)
		GPIO.output(self.en_pin_rf, GPIO.HIGH)

		if self.logger:
			self.logger.debug(f'[Step motor] ACTIVATE pul pin: {self.pul_pin_lf}, dir pin: {self.dir_pin_lf}, en pin: {self.en_pin_lf}')
		return True
		# except Exception as e:
		# 	if self.logger:
		# 		self.logger.debug(f'[Step motor] init error: {e}')
		# 	return False

	# 모터 제어 함수
	def update(self, direction, pulses):
		if direction == 'high':
			GPIO.output(self.dir_pin_lf, GPIO.HIGH)
			GPIO.output(self.dir_pin_rf, GPIO.HIGH)# 방향 설정 (HIGH 또는 LOW)
		else:
			GPIO.output(self.dir_pin_lf, GPIO.LOW)
			GPIO.output(self.dir_pin_rf, GPIO.LOW)

		for i in range(pulses):
			GPIO.output(self.pul_pin_lf, GPIO.HIGH)
			time.sleep(0.0005)  # 펄스 너비
			GPIO.output(self.pul_pin_lf, GPIO.LOW)
			time.sleep(0.0005)  # 펄스 간격
			# GPIO.output(self.pul_pin_rf, GPIO.HIGH)
			# time.sleep(0.0005)
			# GPIO.output(self.pul_pin_rf, GPIO.LOW)

	def upward(self, angle):
		self.update('high', 100)
		if self.logger:
			self.logger.debug(f'[Step motor] upward with angle {100}')

	def downward(self, angle):
		self.update('low', 100)
		if self.logger:
			self.logger.debug(f'[Step motor] downward with angle {100}')

	def terminate(self):
		GPIO.cleanup(self.pul_pin_lf)
		GPIO.cleanup(self.dir_pin_lf)
		GPIO.cleanup(self.en_pin_lf)
		GPIO.cleanup(self.pul_pin_rf)
		GPIO.cleanup(self.dir_pin_rf)
		GPIO.cleanup(self.en_pin_rf)
		if self.logger:
			self.logger.debug(f'[Step motor] terminate Step motor')
