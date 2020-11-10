import torch
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh


class BlobAnnotations(object):
    def __init__(self, h, w):
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, w)
        self.xx, self.yy = np.meshgrid(x, y)
        self.xx = self.xx.flatten()
        self.yy = self.yy.flatten()
        self.xy = np.stack([self.xx, self.yy], axis=-1)

    def __call__(self, pic: torch.Tensor):
        image = pic.numpy()[0, :, :].astype('float64')
        blobs_doh = blob_doh(image, max_sigma=14, threshold=.0002)
        rain_per_blob = []
        for blob in blobs_doh:
            distance = np.sqrt(np.sum(np.power(self.xy - blob[:-1].reshape(1, -1), 2.0), axis=-1))
            x_blob = self.xx[distance < blob[-1]].astype('uint16')
            y_blob = self.yy[distance < blob[-1]].astype('uint16')
            rain_mm = image[x_blob, y_blob].sum()
            rain_per_blob.append(rain_mm)
        annotations = np.concatenate([blobs_doh, np.asarray(rain_per_blob).reshape(-1, 1)], axis=1)
        return pic.float(), torch.from_numpy(annotations).float()
