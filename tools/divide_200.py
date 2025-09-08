import os
import pickle
import numpy as np
import torch
import shutil
from natsort import natsorted

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=10)

path_dir = '../data/test'

make_dir = '../data/test_200'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def log_p(target_path:str):
    folders = np.asarray([int(folder) for folder in natsorted(os.listdir(target_path)) if folder != 'log.p'])
    with open(target_path + '/' + 'log.p', 'wb') as f:
        pickle.dump(folders, f)


createFolder(make_dir)

for sub_dir in os.listdir(path_dir):
    d = os.path.join(path_dir, sub_dir)
    if os.path.isdir(d):
        try:
            for i in range(int(sub_dir), int(sub_dir) + len(os.listdir(d)), 200):
                if i < int(sub_dir) + len(os.listdir(d)) - 200:
                    createFolder(make_dir + '/' + str(i))
        except ValueError:
            pass

dir_list = []
for sub_dir in os.listdir(make_dir):
    dir_list.append(int(sub_dir))
dir_list.sort()
dir_list = np.array(dir_list)

for sub_dir in os.listdir(path_dir):
    print(sub_dir)
    d = os.path.join(path_dir, sub_dir)
    if os.path.isdir(d):
        for file in os.listdir(d):
            for i in range(len(dir_list)):
                try:
                    if int(file[:-2]) < dir_list[i]:
                        from_file_path = os.path.join(path_dir + '/' + sub_dir + '/' + file)
                        to_file_path = os.path.join(make_dir + '/' + str(dir_list[i-1]) + '/' + file)
                        shutil.copyfile(from_file_path, to_file_path)
                        break
                    elif i == (len(dir_list) - 1):
                        from_file_path = os.path.join(path_dir + '/' + sub_dir + '/' + file)
                        to_file_path = os.path.join(make_dir + '/' + str(dir_list[i]) + '/' + file)
                        shutil.copyfile(from_file_path, to_file_path)
                        break
                except ValueError:
                    pass

log_p(make_dir)
