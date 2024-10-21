import Jetson.GPIO as GPIO
import time
import random


class DCControl:
    def __init__(self, logger=None):
        self.logger = logger
        self.left_dir_pin = None
        self.left_pwm_pin = None
        self.right_dir_pin = None
        self.right_pwm_pin = None
        self.pwm_left = None
        self.pwm_right = None

    def init(self, left_dir_pin=5, left_pwm_pin=6, right_dir_pin=20, right_pwm_pin=16):
        try:
            self.left_dir_pin = left_dir_pin
            self.left_pwm_pin = left_pwm_pin
            self.right_dir_pin = right_dir_pin
            self.right_pwm_pin = right_pwm_pin

            # Setup GPIO
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.left_dir_pin, GPIO.OUT)
            GPIO.setup(self.left_pwm_pin, GPIO.OUT)
            GPIO.setup(self.right_dir_pin, GPIO.OUT)
            GPIO.setup(self.right_pwm_pin, GPIO.OUT)

            # Create PWM Instances
            self.pwm_left = GPIO.PWM(self.left_pwm_pin, 1000)
            self.pwm_right = GPIO.PWM(self.right_pwm_pin, 1000)

            # Start PWM with 0% Duty Cycle
            self.pwm_left.start(0)
            self.pwm_right.start(0)

            if self.logger:
                self.logger.info(f'[DC motor] ACTIVATE left dir pin: {self.left_dir_pin}, left pwm pin: {self.left_pwm_pin}, '
                                 f'right dir pin: {self.right_dir_pin}, right pwm pin: {self.right_pwm_pin}')
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f'[DC motor] init error: {e}')
            return False


    def go_forward(self, speed):
        self.update(True, speed, True, speed)
        if self.logger:
            self.logger.info(f'[DC motor] go forward with {speed} speed')

    def turn_left(self, speed):
        self.update(False, 0, True, speed)
        if self.logger:
            self.logger.info(f'[DC motor] turn left with {speed} speed')

    def turn_right(self, speed):
        self.update(True, speed, False, 0)
        if self.logger:
            self.logger.info(f'[DC motor] turn right with {speed} speed')

    def update(self, left_direction, left_duty_cycle, right_direction, right_duty_cycle):
        GPIO.output(self.left_dir_pin, GPIO.HIGH if left_direction else GPIO.LOW)
        self.pwm_left.ChangeDutyCycle(left_duty_cycle)
        GPIO.output(self.right_dir_pin, GPIO.HIGH if right_direction else GPIO.LOW)
        self.pwm_right.ChangeDutyCycle(right_duty_cycle)

    def terminate(self):
        self.pwm_left.stop()
        self.pwm_right.stop()

        GPIO.cleanup(self.left_dir_pin)
        GPIO.cleanup(self.left_pwm_pin)
        GPIO.cleanup(self.right_dir_pin)
        GPIO.cleanup(self.right_pwm_pin)
        if self.logger:
            self.logger.info(f'[DC motor] terminate DC motor')


if __name__ == '__main__':
    dc_controller = DCControl()
    for _ in range(10000):
        # Generate random direction and duty cycle
        left_dir = random.choice([True, False])
        right_dir = random.choice([True, False])
        left_duty = random.uniform(0, 100)
        right_duty = random.uniform(0, 100)

        # Update motors with random values
        dc_controller.update_motor_control(left_dir, left_duty, right_dir, right_duty)

        # Wait for a bit before changing again
        time.sleep(random.uniform(0.5, 5))  # Random delay between 0.5 to 5 seconds

    dc_controller.terminate()
