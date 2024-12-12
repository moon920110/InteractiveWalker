import datetime
import logging
import threading
import serial
import time

from parts.brain import Brain


class Walker:
    def __init__(self):
        self.speed = 50
        self.angle = 0

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



    def _run_imu(self):

            # self.temp = arduino.readline().decode('utf-8')
            # print(self.temp)
        pass

    def _run_brain(self):
        arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)
        while not self.stop_event.is_set():
            arduino.write('7'.encode('utf-8'))
            # self.angle, self.speed = self.brain.think()
            #TODO
            test = 'test'

    def run_walker(self):
        imu_thread = threading.Thread(target=self._run_imu)
        brain_thread = threading.Thread(target=self._run_brain)

        try:
            imu_thread.start()
            self.logger.info(f'[Walker] imu thread start')
            brain_thread.start()
            self.logger.info(f'[Walker] Brain thread start')


            imu_thread.join()
            brain_thread.join()

        except KeyboardInterrupt:
            self.logger.error("[Walker] KeyboardInterrupt")

        finally:
            self.stop_event.set()
            self._terminate()
            self.logger.info("[Walker] terminate Walker")

    def _terminate(self):
        self.brain.terminate()


if __name__ == '__main__':
    walker = Walker()
    walker.run_walker()