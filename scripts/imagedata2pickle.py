import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataset.data_common import preprocess_radar_data, load_radar_image, slice

H = 32
W = 32
n_samples = 10000
image_folder_path = '/data/datasets/radar_maps'
file_list = os.listdir(image_folder_path)
print(len(file_list))
index_list = np.linspace(0, len(file_list) - 1, len(file_list)).astype('uint64')
np.random.shuffle(index_list)  # Select file randomly
image_list = []
for file_index in tqdm(index_list):
    file_path = os.path.join(image_folder_path, file_list[file_index])
    data = load_radar_image(file_path)
    _, row, col = preprocess_radar_data(data, H, W, 5)
    if len(row) > 0:
        for r, c in zip(row, col):
            d = slice(data, r, c, H, W)
            image_list.append(d)
    if len(image_list) > n_samples:
        break
    print(len(image_list))

pickle.dump(image_list, open('/data/datasets/rain_data_val.pickle', 'wb'))
