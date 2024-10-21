import os
import pickle
import numpy as np
import torch
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=10)

path_dir = '../data/train'

BODY_25_pairs = np.array([
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12],
    [12, 13], [13, 14], [1, 0], [14, 15], [15, 16], [14, 17], [11, 18], [18, 19], [11, 20]])

file_len = 0
idx = 0
for sub_dir in os.listdir(path_dir):
    d = os.path.join(path_dir, sub_dir)
    if os.path.isdir(d):
        file_len += len(os.listdir(d))

save_linkloss = np.zeros(shape=(20, file_len))

for sub_dir in os.listdir(path_dir):
    d = os.path.join(path_dir, sub_dir)
    if os.path.isdir(d):
        print(sub_dir)
        for file in os.listdir(d):
            data = pickle.load(open(path_dir + '/' + sub_dir + '/' + file, "rb"))
            data = np.asarray(data)
            keypoint = data[2]
            for i in range(20):
                a = torch.from_numpy(keypoint[BODY_25_pairs[i, 0]])
                b = torch.from_numpy(keypoint[BODY_25_pairs[i, 1]])
                # print(idx)
                save_linkloss[i, idx] = torch.sum((a - b) ** 2).numpy()
            idx += 1
save_linkloss = np.sort(save_linkloss, axis=1)

# pickle.dump()
link_min = np.zeros(shape=20)
link_max = np.zeros(shape=20)
for i in range(20):
    link_min[i] = save_linkloss[i, (file_len * 3) // 100]
    link_max[i] = save_linkloss[i, (file_len * 97) // 100]

with open(path_dir + '/link_min_all.p', 'wb') as file:
    pickle.dump(link_min, file)
with open(path_dir + '/link_max_all.p', 'wb') as file:
    pickle.dump(link_max, file)
# file_list = os.listdir(path_dir)
# print(file_list)