from sensors.sensors import SensorEnv
from model.VisionModel_isaac import FootDetector as isaac_model
from sensors.app.FramerateMonitor import FramerateMonitor
from unreal.env import DogChasingEnv
import numpy as np
import cv2
import time
import os
import pickle
import datetime


def visualize(image):
    if image.dtype != np.uint8:
        image *= 255
        image[image < 0] = 0
        image = image.astype(np.uint8)
    image = cv2.resize(image, (500, 500))
    cv2.imshow("Pressure", image)
    if cv2.waitKey(1) & 0xff == 27:
        return False
    else:
        return True

def init_sensor():
    print("initializing sensors...")
    sensor = SensorEnv(
        ports=["/dev/ttyUSB0"],
        stack_num=20,
        adaptive_calibration=True,
        normalize=True
    )
    print("sensor init finish")
    return sensor

def test_sensor():
    sensor = init_sensor()
    while True:
        images = sensor.get()
        if not visualize(images[-1]):
            break
        #print(images.shape)
        print(f"sensor FPS : {sensor.fps}")
    sensor.close()

def test_model():
    model = isaac_model(visualize=True)
    sensor = init_sensor()
    while True:
        images = sensor.get()
        avail, angle, speed = model(images, hmd_yaw=0)

        visual_image = images[-1]
        if hasattr(model, "visualized_image"):
            visual_image = model.visualized_image
        if not visualize(visual_image):
            break
        print(f"sensor FPS:{sensor.fps}, Avail:{avail}, Angle:{angle}, Speed:{speed}")
    sensor.close()

def main(save_log=False, **kwargs):
    env = DogChasingEnv("127.0.0.1", 13000)
    fpsMonitor = FramerateMonitor()
    model = isaac_model(visualize=True)
    sensor = init_sensor()

    try:
        if save_log:
            now = datetime.datetime.now()
            log_dir = os.path.join(
                kwargs['log_dir'],
                f"{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
            )
            os.makedirs(log_dir, exist_ok=True)

            images_filename = "images.npy"
            infos_filename = "infos.pickle"
            fnp = open(os.path.join(log_dir, images_filename), 'wb')
            fpickle = open(os.path.join(log_dir, infos_filename), 'wb')

        while True:
            images = sensor.get()

            #hmd_yaw = env.get_state()
            hmd_yaw = 0

            avail, angle, speed = model(images, hmd_yaw=hmd_yaw)
            if avail:
                env.move(speed, angle, 1)

            visual_image = images[-1]
            if hasattr(model, "visualized_image"):
                visual_image = model.visualized_image
            if not visualize(visual_image):
                break

            main_fps = round(fpsMonitor.getFps())
            sensor_fps = sensor.fps
            print(f"sensor FPS : {sensor_fps}, main FPS: {main_fps}, Speed: {speed}, Angle: {angle}")
            fpsMonitor.tick()

            if save_log:
                np.save(fnp, images[-1])

                data = {
                    "sensor_fps": sensor_fps,
                    "main_fps": main_fps,
                    "avail": avail,
                    "speed": speed,
                    "angle": angle,
                    "hmd_yaw": hmd_yaw,
                    "time": time.time()
                }
                pickle.dump(data, fpickle, protocol=pickle.HIGHEST_PROTOCOL)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        sensor.close()
        if save_log:
            fnp.close()
            fpickle.close()

if __name__ == "__main__":
    #main(save_log=True, log_dir=".\\logs")
    #test_model()
    test_sensor()

