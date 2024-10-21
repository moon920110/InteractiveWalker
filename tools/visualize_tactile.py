import numpy as np
import h5py
import os
import cv2


input_data_path = './2023-08-02-1_test_carpets/2023-08-02_12-26-53_testing_carpets_squat/'

for sub_dir in os.listdir(input_data_path):
    d = os.path.join(input_data_path, sub_dir)

    if d.endswith("data.hdf5"):
        with h5py.File(d, 'r') as f:
            tile_list = ['tactile-carpet-c', 'tactile-carpet-d', 'tactile-carpet-left', 'tactile-carpet-right']
            c = f['tactile-carpet-c']['tactile_data']['data']
            d = f['tactile-carpet-d']['tactile_data']['data']
            l = f['tactile-carpet-left']['tactile_data']['data']
            r = f['tactile-carpet-right']['tactile_data']['data']
            cd = np.append(c, d, axis=2)
            lr = np.append(l, r, axis=2)
            cd_lr = np.append(cd, lr, axis=1)
            print(cd_lr.shape)
            for i in range(cd_lr.shape[0]):
                cv2.imshow('frame', cv2.resize(cd_lr[i]/10/255, (600,600)))
                cv2.waitKey(100)
                print(i)
            cv2.destroyAllWindows()
