import os
import shutil
from distutils.dir_util import copy_tree
import numpy as np
import pickle
from natsort import natsorted
import glob
import argparse
import random

parser = argparse.ArgumentParser(description='making tmp directory')
parser.add_argument('--lunge', type=bool, default=False)
parser.add_argument('--squat', type=bool, default=False)
parser.add_argument('--all', type=bool, default=False)
args = parser.parse_args()



def train_test(exercise:str):
    train_dir = '../data/' + exercise + '_train_1125_HY/'
    test_dir = '../data/' + exercise + '_test_1125_HY/'

    experiment_all = [experiment + '/tmp_train' for experiment in natsorted(glob.glob('../data/*'))
                      if experiment.split('_')[-2] == exercise and experiment.split('_')[-3] != 'EH']
    print(experiment_all, '\n')
    experiment_test = [experiment for experiment in experiment_all if experiment.split('/')[2].split('_')[0] == '2022-11-25' and experiment.split('/')[-2].split('_')[-3] == 'HY']
    #experiment_test = [experiment_all[i] for i in test_idx if experiment_all[i].split('/')[2].split('_')[0] == '2022-11-25']
    experiment_train = natsorted(list(set(experiment_all) - set(experiment_test)))
    experiment_train = [x for x in experiment_train if x.split('/')[2].split('_')[0] == '2022-11-25']
    print(experiment_test, '\n')
    print(experiment_train)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return experiment_train, experiment_test, train_dir, test_dir



def data_split(exercise:str, train:int, val:int, test:int, seed:int):

    random.seed(seed)

    experiment_all = [experiment + '/tmp_train'
                      for experiment in natsorted(glob.glob('../data/*'))]

    target_experiment = [experiment for experiment in experiment_all
                         if experiment.split('/')[-2].split('_')[-2] == exercise and os.path.exists(experiment)]

    train_data = random.sample(target_experiment, train)
    val_data = random.sample(target_experiment, val)
    test_data = random.sample(target_experiment, test)

    return train_data, val_data, test_data



def copy_p(source_path:list, target_path:str):
    sum_len = 0
    for experiment in source_path:
        copy_tree(experiment, target_path + '/' + str(sum_len))
        new_path = target_path + '/' + str(sum_len) + '/'
        sum_len += len(os.listdir(new_path))

    sum_len = 0
    for sub_folder in natsorted(os.listdir(target_path)):
        print(sub_folder)
        len_file = len(os.listdir(target_path + '/' + sub_folder))
        for file in os.listdir(target_path + '/' + sub_folder):
            os.rename(target_path + '/' + sub_folder + '/' + file, target_path + '/' + sub_folder + '/' + f'{500000+sum_len + int(file[:-2])}.p')
        for file in os.listdir(target_path + '/' + sub_folder):
            os.rename(target_path + '/' + sub_folder +'/' + file, target_path + '/' + sub_folder + '/' + f'{int(file[:-2]) - 500000}.p')
        sum_len += len_file



def log_p(target_path:str):
    folders = np.asarray([int(folder) for folder in natsorted(os.listdir(target_path)) if folder != 'log.p'])
    with open(target_path + '/' + 'log.p', 'wb') as f:
        pickle.dump(folders, f)


if args.lunge:
    seed=10
    train_path = '../data/lunge_train'
    val_path = '../data/lunge_val'
    test_path = '../data/lunge_test'

    train_data, val_data, test_data = data_split('lunge', train=9, val=3, test=2, seed=seed)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print('Copying files')
    print('train_data : ', train_data, '\n', '->', train_path)
    copy_p(train_data, train_path)
    print('val_data : ', val_data, '\n', '->', val_path)
    copy_p(val_data, val_path)
    print('test_data : ', test_data, '\n', '->', test_path)
    copy_p(test_data, test_path)
    print('Writing log.p')
    log_p(train_path)
    log_p(val_path)
    log_p(test_path)



if args.all:
    seed = 10
    train_path = '../data/train'
    val_path = '../data/val'
    test_path = '../data/test'

    squat_train, squat_val, squat_test = data_split('squat', train=13, val=3, test=3, seed=seed)
    lunge_train, lunge_val, lunge_test = data_split('lunge', train=9, val=3, test=2, seed=seed)
    crunch_train, crunch_val, crunch_test = data_split('crunch', train=9, val=3, test=2, seed=seed)

    train_data = squat_train + lunge_train + crunch_train
    val_data = squat_val + lunge_val + crunch_val
    test_data = squat_test + lunge_test + crunch_test

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print('Copying files')
    print('train_data : ', train_data, '\n', '->', train_path)
    copy_p(train_data, train_path)
    print('val_data : ', val_data, '\n', '->', val_path)
    copy_p(val_data, val_path)
    print('test_data : ', test_data, '\n', '->', test_path)
    copy_p(test_data, test_path)
    print('Writing log.p')
    log_p(train_path)
    log_p(val_path)
    log_p(test_path)