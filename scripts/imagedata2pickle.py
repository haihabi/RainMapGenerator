import os
import pickle
import numpy as np
from tqdm import tqdm
from dataset.data_common import preprocess_radar_data, load_radar_image, slice


def save_image2pickle(h, w, image_folder_path, n_samples, r=0.1, labeled=False):
    file_list = os.listdir(image_folder_path)
    print(len(file_list))

    index_list = np.linspace(0, len(file_list) - 1, len(file_list)).astype('uint64')
    np.random.shuffle(index_list)  # Select file randomly
    image_list = []
    for file_index in tqdm(index_list):
        file_path = os.path.join(image_folder_path, file_list[file_index])
        data = load_radar_image(file_path)
        _, row, col = preprocess_radar_data(data, h, h, 5)
        if len(row) > 0:
            for r, c in zip(row, col):
                d = slice(data, r, c, h, w)
                dn = d / np.max(d)
                rain_pixels = np.sum(dn > r) / (h * w)
                if rain_pixels > r:
                    if labeled:
                        image_anno = (rain_pixels)
                        image_list.append((d, image_anno))
                    else:
                        image_list.append(d)
        if len(image_list) > n_samples:
            break
    return image_list


H = 32
W = 32
n_s = 10000
data_folder_path = '/data/datasets/radar_maps'

labled = False

out_list = save_image2pickle(H, W, data_folder_path, 40000, labeled=False)
pickle.dump(out_list, open('/data/datasets/rain_data_val_large.pickle', 'wb'))

out_list = save_image2pickle(H, W, data_folder_path, 400000, labeled=False)
pickle.dump(out_list, open('/data/datasets/rain_data_large.pickle', 'wb'))
