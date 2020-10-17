import numpy as np
import random
from dataset.data_common import read_radar_files, load_radar_image, preprocess_radar_data, sample_slice
from torch.utils.data import Dataset


class RadarStaticDataSet(Dataset):
    def __init__(self, base_folder: str, h: int, w: int, n_samples4epoch=50000, n_total_samples=100000, min_points=10,
                 download=False, pickle=False):
        self.base_folder = base_folder
        self.data_dict = dict()
        self.data_dict_keys = list(self.data_dict.keys())
        self.file_dict = read_radar_files(base_folder)
        self.file_list = list(self.file_dict.values())
        self.n_total_samples = n_total_samples
        self.n_samples4epoch = n_samples4epoch
        self.h = h
        self.w = w
        self.n_samples = 0
        self.min_points = min_points
        if download:
            pass
        print("The total number of files:" + str(len(self.file_list)))

    def __len__(self):
        return self.n_samples4epoch

    def __getitem__(self, index):
        return self.get_sample()

    def get_sample(self):
        index = int(np.random.randint(0, len(self.file_list), 1).astype('int'))
        if self.data_dict.get(index) is None and not self.n_samples >= self.n_total_samples:
            r = []
            while len(r) <= 1:
                data_full = load_radar_image(self.file_list[index]).astype('float32')
                data_full, r, c = preprocess_radar_data(data_full, self.h, self.w, self.min_points)
                if len(r) <= 1:
                    index = int(np.random.randint(0, len(self.file_list), 1).astype('int'))

            self.n_samples += len(r)
            self.data_dict.update({index: (data_full, r, c)})
            self.data_dict_keys = list(self.data_dict.keys())
        else:
            index = random.choice(self.data_dict_keys)
            data_full, r, c = self.data_dict.get(index)
        d = sample_slice(data_full, r, c, self.h, self.w)
        return d

    def number_of_files_load(self):
        return len(self.data_dict)
