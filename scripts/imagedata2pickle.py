import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataset.data_common import preprocess_radar_data, load_radar_image, slice


class AnnotationInit(object):
    def __init__(self, h, w):
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, w)
        self.xx, self.yy = np.meshgrid(x, y)
        self.xx = self.xx.flatten()
        self.yy = self.yy.flatten()
        self.xy = np.stack([self.xx, self.yy], axis=-1)


def image_annotation(image, anno_init: AnnotationInit):
    blobs_doh = blob_doh(image, max_sigma=14, threshold=.0002)
    rain_per_blob = []
    for blob in blobs_doh:
        distance = np.sqrt(np.sum(np.power(anno_init.xy - blob[:-1].reshape(1, -1), 2.0), axis=-1))
        x_blob = anno_init.xx[distance < blob[-1]].astype('uint16')
        y_blob = anno_init.yy[distance < blob[-1]].astype('uint16')
        rain_mm = image[x_blob, y_blob].mean()
        rain_per_blob.append(rain_mm)
    return np.concatenate([blobs_doh, np.asarray(rain_per_blob).reshape(-1, 1)], axis=1)


def save_image2pickle(h, w, image_folder_path, n_samples, anno_init=None):
    file_list = os.listdir(image_folder_path)
    print(len(file_list))
    labeled = anno_init is not None
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
                if labeled:
                    image_anno = image_annotation(d, anno_init)
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
if labled:
    anno_init = AnnotationInit(H, W)
else:
    anno_init = None
out_list = save_image2pickle(H, W, data_folder_path, 40000, anno_init=anno_init)
pickle.dump(out_list, open('/data/datasets/rain_data_val_large.pickle', 'wb'))

out_list = save_image2pickle(H, W, data_folder_path, 400000, anno_init=anno_init)
pickle.dump(out_list, open('/data/datasets/rain_data_large.pickle', 'wb'))
