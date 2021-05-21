from torchvision import transforms
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization
from matplotlib import pyplot as plt
import numpy as np
import torch

training_data_pickle = '/Users/haihabi/data/rain_data_val.pickle'
transform_training = transforms.Compose([
    transforms.ToTensor(),
    MaxNormalization(),
])
# TODO: add data augmentation
train_rds = RadarDataSet(training_data_pickle, transform=transform_training)
k = 13
for i in range(len(train_rds)):
    image, ann = train_rds[i]
    pool = torch.nn.MaxPool2d(k, stride=1, padding=int(k / 2))
    image_p = pool(image)
    image_eq = (image_p == image) * (image_p != 0)
    image_eq = image_eq.numpy()[0, :, :].astype('float64')
    image = image.numpy()[0, :, :].astype('float64')
    image = image / np.max(image)
    rain_rate = np.sum(image > 0.1) / (32 * 32)
    if np.sum(image > 0.1) / (32 * 32) > 0.1:
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(image_eq)
        plt.show()
    # # if i > 0:
    # #     break
    # # print("a")
