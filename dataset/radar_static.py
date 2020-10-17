from dataset.data_common import radar2rain
from torch.utils.data import Dataset
import pickle


class RadarDataSet(Dataset):
    def __init__(self, pickle_path: str, transform=None):
        self.pickle_path = pickle_path
        self.pickle_data = pickle.load(open(pickle_path, 'rb'))
        self.data_shape = self.pickle_path[0].shape
        self.data_preprocess = dict()
        self.n_samples = len(self.pickle_data)
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        data = self.get_sample(index)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_sample(self, index):
        data_image = self.data_preprocess.get(index)
        if data_image is None:
            sample = self.pickle_data[index]
            data_image = radar2rain(sample)
            self.data_preprocess.update({index: data_image})
        return data_image

    def number_of_files_load(self):
        return len(self.data_dict)
