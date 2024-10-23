from sensors.sensors import MultiSensors
from sensors.storage.Storage import createStorage
import copy
from sensors.app.AppContext import AppContext
from sensors.common.dataset_tools import *
import numpy as np
from PIL import Image
from time import time, sleep


def main(
        max_frame = 100,
        foldername = '/home/cilab/media/yhssd',
        filename = 'test',
        normalize = False,
        counter = 0,
        norm_img_list = []
    ):

    sensor = MultiSensors(['/dev/ttyUSB0'])
    print("initializing sensors...")
    sensor.init_sensors()
    print("initializing sensors...Done")
    storage = createStorage('hdf5', foldername, filename, AppContext.create(), {'blockSize': 90})
    base_images = []
    start_signal = 1
    base_time = time()
    print('calibration done! collection strat at ', base_time)

    while storage.frameCount < max_frame:

        if start_signal == 1:
            start_signal = 0
            for i in range(20):
                total_image = sensor.get()
                base_images.append(total_image)
            base_images = np.array(base_images)
            base_image = np.mean(base_images, axis=0)
            print(total_image.shape)


        total_image = sensor.get()
        total_image = total_image - base_image

        #visualize
        visual_image = copy.deepcopy(total_image)
        visual_image *= 255
        visual_image = visual_image.astype(np.uint8)
        visual_image = cv2.resize(visual_image, (500, 500))
        total_image2 = copy.deepcopy(total_image)

        total_image2 /= 1500
        total_image2 = total_image2 * 255
        total_image2 = np.clip(total_image2, 0 ,255)
        # print(total_image2)

        total_image2 = cv2.resize(total_image2.astype(np.uint8), (500, 500))

        cv2.imshow("Pressure", total_image2)
        if cv2.waitKey(1) & 0xff == 27:
            break

        #fps
        fps = sensor.getFps()

        #unix timestep
        ts = getUnixTimestamp()

        #store data
        storage.addFrame(ts, {'pressure': total_image})

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
        foldername = f'walker_data1024/{name}',
        filename = label,
        normalize = False
    )