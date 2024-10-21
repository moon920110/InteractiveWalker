import torch
import h5py
import os
import numpy as np
from torch.utils.data import Dataset
import natsort
import pickle
import pandas as pd
import copy
import re
from scipy.ndimage import rotate, shift
import random
import argparse
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing



def rotated(data):
    random_angle = random.randint(-5, 5)
    data = np.array([rotate(image, random_angle, reshape=False, mode='nearest') for image in data])
    return data

def shifted(data):
    random_x = random.randint(-5, 5)
    random_y = random.randint(-5, 5)
    random_vector = (random_x, random_y)
    data = np.array([shift(image, random_vector, mode='nearest') for image in data])
    return data

def resized(data):

    return data

def time_scaled(data):
    #if it need
    return data

def augmentation(args):
    results = []
    for filename in os.listdir(args.data_path):
        if filename == '.DS_Store': continue
        if filename == '\x00': continue

        with open(args.data_path + filename, 'rb') as f:
            data = pickle.load(f)
        data['acc'] = data['acc'] - np.array(data['acc'][:10]).mean() # initial calibration
        temp = pd.DataFrame(data['acc']) # acc smoothing
        temp = temp.rolling(15, center=True, axis=0).mean()
        # temp = temp.dropna()
        test_data_tactile = data['tactile'][-999:-499]
        test_data_acc = list(temp[-999:-499].values)
        data['tactile'] = data['tactile'][500:-999]
        data['acc'] = list(temp[500:-999].values)


        # with open(args.save_path + filename + '_origin', 'wb') as f:
        #     pickle.dump(data, f)
        acc_label = []
        test_acc_label = []
        for i in range(args.epoch):
            temp_data = data['acc']
            test_temp_data = test_data_acc
            for temp_acc in temp_data:
                if temp_acc >= 0.02:
                    acc_label.append([1, 0, 0])
                elif temp_acc <= -0.05:
                    acc_label.append([0, 0, 1])
                else: acc_label.append([0, 1, 0])
            for temp_acc in test_temp_data:
                if temp_acc >= 0.02:
                    test_acc_label.append([1, 0, 0])
                elif temp_acc <= -0.05:
                    test_acc_label.append([0, 0, 1])
                else: test_acc_label.append([0, 1, 0])
            # temp_data = rotated(temp_data)
            # temp_data = shifted(temp_data)
            concat_data = [data['tactile'], temp_data, acc_label]
            test_concat_data = [test_data_tactile, test_temp_data, test_acc_label]
            with open(args.save_path + filename + '_' + str(i), 'wb') as f:
                pickle.dump(concat_data, f)
            with open(args.save_path + 'test_' +filename + '_' + str(i), 'wb') as f:
                pickle.dump(test_concat_data, f)

    return results

def generate_test_dataset(args):
    for filename in os.listdir(args.data_path):
        if filename == '.DS_Store': continue

        if 'AMS' in filename:
            with open(args.data_path + filename, 'rb') as f:
                data = pickle.load(f)
            if len(data[0]) != len(data[4]):
                min_len = min(len(data[0]), len(data[4]))
                data[0] = data[0][:min_len]
                data[1] = data[1][:min_len]
                data[2] = data[2][:min_len]
                data[3] = data[3][:min_len]
                data[4] = data[4][:min_len]
            temp = pd.DataFrame(data[2]) # emg smoothing
            temp = temp.rolling(30, center=True, axis=0).mean()
            temp = temp.dropna()
            data[2] = temp.to_numpy()
            data[0] = data[0][15:-14]
            data[1] = data[1][15:-14]
            data[3] = data[3][15:-14]
            data[4] = data[4][15:-14]

            # with open(args.save_path + filename + '_origin', 'wb') as f:
            #     pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tactile2EMG data augmentation code')
    parser.add_argument('--data_path', type=str, default='../data/original_data/', help='Experiment path')
    parser.add_argument('--epoch', type=int, default=1, help='total epoch')
    parser.add_argument('--save_path', type=str, default='../data/smoothed_data_cut_labels_splitTest/', help='save path')
    # parser.add_argument('--save_path', type=str, default='/media/jjyunho/YH_SSD/(2024.03.26)_lunge_data/', help='save path')
    args = parser.parse_args()

    augmentation(args)
    # generate_test_dataset(args)