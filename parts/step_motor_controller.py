import Jetson.GPIO as GPIO
import time


class StepMotorControl:
	def __init__(self, logger=None):
		self.logger = logger
		self.pul_pin = None
		self.dir_pin = None
		self.en_pin = None

	def init(self, pul_pin=5, dir_pin=6, en_pin=7):
		try:
			self.pul_pin = pul_pin
			self.dir_pin = dir_pin
			self.en_pin = en_pin

			GPIO.setmode(GPIO.BOARD)
			GPIO.setup(self.pul_pin, GPIO.OUT)
			GPIO.setup(self.dir_pin, GPIO.OUT)
			GPIO.setup(self.en_pin, GPIO.OUT)

			GPIO.output(self.en_pin, GPIO.LOW)

			if self.logger:
				self.logger.Debug(f'[Step motor] ACTIVATE pul pin: {self.pul_pin}, dir pin: {self.dir_pin}, en pin: {self.en_pin}')
			return True
		except Exception as e:
			if self.logger:
				self.logger.Error(f'[Step motor] init error: {e}')
			return False

	# 모터 제어 함수
	def update(self, direction, pulses):
		if direction == 'high':
			GPIO.output(self.dir_pin, GPIO.HIGH)  # 방향 설정 (HIGH 또는 LOW)
		else:
			GPIO.output(self.dir_pin, GPIO.LOW)

		for i in range(pulses):
			GPIO.output(self.pul_pin, GPIO.HIGH)
			time.sleep(0.0005)  # 펄스 너비
			GPIO.output(self.pul_pin, GPIO.LOW)
			time.sleep(0.0005)  # 펄스 간격

	def upward(self, angle):
		self.update('high', angle)
		if self.logger:
			self.logger.Debug(f'[Step motor] upward with angle {angle}')

	def downward(self, angle):
		self.update('low', angle)
		if self.logger:
			self.logger.Debug(f'[Step motor] downward with angle {angle}')

	def terminate(self):
		GPIO.cleanup(self.pul_pin)
		GPIO.cleanup(self.dir_pin)
		GPIO.cleanup(self.en_pin)
		if self.logger:
			self.logger.Debug(f'[Step motor] terminate Step motor')
