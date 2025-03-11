import datetime
import logging
import threading
import serial
import time
from getkey import getkey
import queue

from parts.brain import Brain


class Walker:
    def __init__(self):
        self.forback = 0
        self.leftright = 0
        self.STS = False
        self.isStand = True
        self.STS_flag = False
        self.leftright_flag = False

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



    def _run_imu(self, keyQueue):
        arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)
        while not self.stop_event.is_set():
            if keyQueue.empty:
                pass
            else: print('key: ', keyQueue)

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
            else:
                command = 'S1 ' + str(int(1000 * self.forback)) + ',S2 ' + str(int(1000 * self.forback)) + ",D1 0,D2 0,"
                # arduino.write('S1 0,S2 0,D1 0,D2 0'.encode('utf-8'))
                arduino.write(command.encode('utf-8'))
                # print(self.angle, self.speed, 'write')
                time.sleep(0.1)
                # self.temp = arduino.readline().decode('utf-8')
                # print(self.temp)
            # print(command)
        pass

    def _run_brain(self):
        while not self.stop_event.is_set():
            self.forback, self.leftright, self.STS = self.brain.think()
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

    def _run_keyinput(self, keyQueue):
        while not self.stop_event.is_set():
            key = input()
            keyQueue.put(key)

    def run_walker(self):
        keyQueue = queue.Queue()
        imu_thread = threading.Thread(target=self._run_imu, args=(keyQueue,))
        brain_thread = threading.Thread(target=self._run_brain)
        keyinput_thread = threading.Thread(target=self._run_keyinput, args=(keyQueue,))

        try:
            imu_thread.start()
            self.logger.info(f'[Walker] imu thread start')
            brain_thread.start()
            self.logger.info(f'[Walker] Brain thread start')
            keyinput_thread.start()
            self.logger.info(f'[Walker] KeyInput thread start')


            imu_thread.join()
            brain_thread.join()
            keyinput_thread.join()

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