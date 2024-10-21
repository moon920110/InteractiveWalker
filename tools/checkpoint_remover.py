import os
import glob

all_checkpoints = [file for file in glob.glob('../data/2022-11-09_19-58-37_testing_carpets_JH_squat_1/ckpts/*tar')]
best_checkpoints = [file for file in glob.glob('../data/2022-11-09_19-58-37_testing_carpets_JH_squat_1/ckpts/*best.path.tar')]
remove_checkpoints = list(set(all_checkpoints) - set(best_checkpoints))

for checkpoint in remove_checkpoints:
    os.remove(checkpoint)