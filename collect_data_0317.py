from tactile_collecting.sensors.sensors import MultiSensors
from tactile_collecting.sensors.storage.Storage import createStorage
import copy
from tactile_collecting.sensors.app.AppContext import AppContext
from tactile_collecting.sensors.common.dataset_tools import *
import numpy as np
from time import time, sleep
import serial
from multiprocessing import Manager


def main(
        max_frame = 100,
        foldername = '/home/cilab/media/yhssd',
        filename = 'test',
        normalize = False,
        counter = 0,
        norm_img_list = []
    ):
    manager = Manager()
    stage = manager.Queue(1)

    stage.put('initialize')

    sensor = MultiSensors(['/dev/ttyUSB0'], stage)
    print("initializing sensors...")
    sensor.init_sensors()
    print("initializing sensors...Done")
    storage = createStorage('hdf5', foldername, filename, AppContext.create(), {'blockSize': 90})
    base_images = []
    start_signal = 1
    base_time = time()
    print('calibration done! collection strat at ', base_time)
    arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)
    bad_row_indexs = [16, 27]
    bad_col_indexs = [9]

    while storage.frameCount < max_frame:

        # if start_signal == 1:
        #     start_signal = 0
        #     print('get ready')
        #     for i in range(30):
        #         pass
        #     print('Initial arm data collecting')
        #     for i in range(50):
        #         total_image = sensor.get()
        #         base_images.append(total_image)
        #     base_images = np.array(base_images)
        #     base_image = np.mean(base_images, axis=0)
        #     print(total_image.shape)

        print(storage.frameCount)
        print('stage at main code does empty? :', stage.empty())
        if storage.frameCount == 50:
            # stage.put('collect')
            print('tactile signal mode changed to collect')


        total_image = sensor.get()
        print('total image:', total_image.shape)
        for row_index in bad_row_indexs:
            prev_row = total_image[row_index - 1,:].astype(np.float32)
            next_row = total_image[row_index + 1,:].astype(np.float32)
            print('row:',((prev_row + next_row) / 2).astype(np.float32))
            total_image[row_index,:] = ((prev_row + next_row) / 2).astype(np.float32)

        for col_index in bad_col_indexs:
            prev_col = total_image[:, col_index - 1].astype(np.float32)
            next_col = total_image[:, col_index + 1].astype(np.float32)
            print('col:', ((prev_col + next_col) / 2).astype(np.float32))
            total_image[:, col_index] = ((prev_col + next_col) / 2).astype(np.float32)

        base_base_image = np.full(total_image.shape, 4096) - total_image
        base3_image = base_base_image - np.min(base_base_image)
        total_image  = np.clip(((base3_image/np.max(base3_image)) * 255), 0, 255)

        # total_image = total_image #- base_image
        total_image = total_image.astype(np.float64)  # - base_image

        #visualize
        visual_image = copy.deepcopy(total_image)
        # visual_image *= 255
        visual_image = visual_image.astype(np.uint8)
        visual_image = cv2.resize(visual_image, (500, 500))
        # total_image2 = copy.deepcopy(total_image)
        #
        # total_image2 /= 1500.0
        # total_image2 = total_image2 * 255
        # total_image2 = np.clip(total_image2, 0 ,255)
        # print(total_image2)

        # total_image2 = cv2.resize(visual_image.astype(np.uint8), (500, 500))

        cv2.imshow("Pressure", visual_image)
        if cv2.waitKey(1) & 0xff == 27:
            break

        #fps
        fps = sensor.getFps()

        #unix timestep
        ts = getUnixTimestamp()

        #imu
        # arduino.write('imu'.encode('utf-8'))
        # sleep(0.03)
        # imu = float(arduino.readline().decode('utf-8').strip())
        imu = 0
        print(imu)

        #store data
        storage.addFrame(ts, {'pressure': total_image, 'imu': imu})

        #verbose
        print(f"FPS : {fps}, time: {time()}, Frames : {storage.frameCount}, Storage : {foldername}/{storage.getName()}")
        # sleep(2)

    sensor.close()

if __name__ == "__main__":
    print("data collection start")
    name = input("Enter your name :").strip()
    label = input("Enter label :").strip()
    max_frame = int(input("Enter max frame :").strip())

    main(
        max_frame = max_frame,
        foldername = f'./data_collect/{name}',
        filename = label,
        normalize = False
    )