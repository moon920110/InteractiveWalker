import os
from glob import glob
from natsort import natsorted
import argparse

parser = argparse.ArgumentParser(description='Run preprocessing')
parser.add_argument('--trim_video', type=bool, default=False, help='Set True when you trim video')
parser.add_argument('--read_keypoints', type=bool, default=False, help='Set True when you read keypoints')
parser.add_argument('--trainable', type=bool, default=False, help='Set True when you generate trainable pickle files')
args = parser.parse_args()

experiments = natsorted(glob('../data/*'))
for experiment in experiments:
    try:
        if args.trim_video:
            print('Trimming video')
            print(experiment)
            os.system('python preprocessing.py --path {} --trim_video True'.format(experiment))
        if args.read_keypoints:
            print('Reading keypoints')
            print(experiment)
            os.system('python preprocessing.py --path {} --read_keypoints True'.format(experiment))
        if args.trainable:
            print('Generate tmp pickle files')
            print(experiment)
            os.system('python preprocessing.py --path {} --trainable True'.format(experiment))
    except FileNotFoundError:
        print(experiment)