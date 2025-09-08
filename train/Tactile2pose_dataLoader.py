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

class tactile2emg_data(Dataset):
    def __init__(self, data):
        self.data = data[0]
        self.label = data[1]

    def __len__(self):
        return self.label[:-20].shape[0]

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx:idx+20])
        label = torch.from_numpy(self.label[idx+20])
        return data, label

    def normalize(self, normalize):
        if normalize == 'minmax':
            self.data = self.data / 1000

def rotated(data):
    random_angle = random.randint(-10, 10)
    data = np.array([rotate(image, random_angle, reshape=False, mode='reflect') for image in data])
    return data

def shifted(data):
    random_x = random.randint(-5, 5)
    random_y = random.randint(-5, 5)
    random_vector = (random_x, random_y)
    data = np.array([shift(image, random_vector, mode='reflect') for image in data])
    return data

def resized(data):

    return data

def time_scaled(data):
    #if it need
    return data

def load_all_sequences(path, test_list, is_test, train_exercise):
    results = []
    for filename in os.listdir(path):
        if filename == '.DS_Store': continue
        if filename[:3] in test_list and is_test == False:
            continue
        elif filename[:3] not in test_list and is_test == True:
            continue
        # elif 'origin' not in filename and is_test == True: # for augmented data
        #     continue
        if train_exercise == 'all':
            pass
        else:
            if train_exercise not in filename:
                continue


        with open(path + filename, 'rb') as f:
            data = pickle.load(f)
            class_label = filename[4:9]


        result = {
            "tactile": data[1].reshape(-1, 1, 64, 64),
            "pose": data[4],
            "class_label": filename
        }
        results.append(result)
    return results


class ExerciseDataset(Dataset):
    def __init__(self, path, train_exercise, window_size, test_list, is_test):
        # self.origin_sequences = []

        self.origin_sequences = load_all_sequences(path, test_list, is_test, train_exercise)
        print(f"Loaded {path}")
        self.is_test = is_test
        self.window_size = window_size
        self.input_sequences = copy.deepcopy(self.origin_sequences)

        self.seq_indexs = []
        start = 0
        for i, seq in enumerate(self.input_sequences):
            end = start + len(seq["tactile"]) - (self.window_size -1)
            self.seq_indexs.append((i, start, end))
            start = end

    def __len__(self):
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        for i, start, end in self.seq_indexs:
            if idx >= start and idx < end:
                real_idx = idx - start

                data = self.input_sequences[i]["tactile"][real_idx:real_idx+self.window_size]
                label = self.input_sequences[i]["pose"][real_idx+self.window_size-1]
                break
        data = data.squeeze(1)
        # if self.is_test == False:
        #     data = rotated(data)
        #     data = shifted(data)
        # data = resized(data)
        # data = np.expand_dims(data, axis=1)
        # return torch.FloatTensor(data), torch.FloatTensor(np.array(label[5])) # for single(abs) muscle
        return torch.FloatTensor(data), torch.FloatTensor(np.array(label)) # for all muscle