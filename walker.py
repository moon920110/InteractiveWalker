import datetime
import logging
import threading
import serial
import time

from parts.DC_motor_controller import DCControl
from parts.step_motor_controller import StepMotorControl
from parts.brain import Brain


class Walker:
    def __init__(self):
        self.speed = 0
        self.angle = 0

        # TODO: IMU
        self.tilt = 0

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        fh = logging.FileHandler(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # self.dc_motor = DCControl(logger=self.logger)
        # self.step_motor = StepMotorControl(logger=self.logger)
        self.brain = Brain(logger=self.logger)

        self.stop_event = threading.Event()

        try:
            self.init()
        except Exception as e:
            self.logger.error(f"[Walker] init error: {e}")
            exit(1)

    def init(self):
        # dc_check = self.dc_motor.init()
        # step_check = self.step_motor.init()
        brain_check = self.brain.init()

        # if dc_check and step_check and brain_check:
        #     self.logger.info("[Walker] init finish")
        #     return True
        # else:
        #     self.logger.error("[Walker] init error")
        #     return False

    def _run_imu(self):
        arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)
        while not self.stop_event.is_set():
            arduino.write('7'.encode('utf-8'))
            time.sleep(0.05)
            self.tilt = arduino.readline()
            print(self.tilt)
        pass

    def _run_dc(self):
        while not self.stop_event.is_set():
            if self.angle > 45:
                self.dc_motor.turn_right(self.speed)
            elif self.angle < -45:
                self.dc_motor.turn_left(self.speed)
            else:
                self.dc_motor.go_forward(self.speed)

    def _run_step(self):
        while not self.stop_event.is_set():
            if self.tilt > 45:
                self.step_motor.upward(self.tilt)
            elif self.tilt < -45:
                self.step_motor.downward(self.tilt)

    def _run_brain(self):
        while not self.stop_event.is_set():
            self.angle, self.speed = self.brain.think()

    def run_walker(self):
        imu_thread = threading.Thread(target=self._run_imu)
        # dc_thread = threading.Thread(target=self._run_dc)
        # step_thread = threading.Thread(target=self._run_step)
        brain_thread = threading.Thread(target=self._run_brain)

        try:
            imu_thread.start()
            self.logger.info(f'[Walker] imu thread start')
            brain_thread.start()
            self.logger.info(f'[Walker] Brain thread start')
            # dc_thread.start()
            # self.logger.info(f'[Walker] DC motor thread start')
            # step_thread.start()
            # self.logger.info(f'[Walker] Step motor thread start')

            imu_thread.join()
            # step_thread.join()
            # dc_thread.join()
            brain_thread.join()

        except KeyboardInterrupt:
            self.logger.error("[Walker] KeyboardInterrupt")

        finally:
            self.stop_event.set()
            self._terminate()
            self.logger.info("[Walker] terminate Walker")

    def _terminate(self):
        # self.dc_motor.terminate()
        # self.step_motor.terminate()
        self.brain.terminate()


if __name__ == '__main__':
    walker = Walker()
    walker.run_walker()