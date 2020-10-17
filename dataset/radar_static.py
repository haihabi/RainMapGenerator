from dataset.data_common import radar2rain
from torch.utils.data import Dataset
import pickle


class RadarDataSet(Dataset):
    def __init__(self, pickle_path: str):
        self.pickle_path = pickle_path
        self.pickle_data = pickle.load(open(pickle_path, 'rb'))
        self.data_preprocess = dict()
        self.n_samples = len(self.pickle_data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index):
        data_image = self.data_preprocess.get(index)
        if data_image is None:
            sample = self.pickle_data[index]
            data_image = radar2rain(sample)
            self.data_preprocess.update({index: data_image})
        return data_image

    def number_of_files_load(self):
        return len(self.data_dict)
