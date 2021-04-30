from torchvision import transforms
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization
from analysis.blob_annotations import BlobAnnotations
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt

training_data_pickle = '/data/datasets/rain_data.pickle'
transform_training = transforms.Compose([
    transforms.ToTensor(),
    MaxNormalization(),
    BlobAnnotations(32, 32),
])
# TODO: add data augmentation
train_rds = RadarDataSet(training_data_pickle, transform=transform_training)
for i in range(len(train_rds)):
    image, ann = train_rds[i]
    image = image.numpy()[0, :, :].astype('float64')
    blobs_log = blob_log(image, max_sigma=30, num_sigma=10, threshold=.03)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    blobs_dog = blob_dog(image, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image, max_sigma=14, threshold=.0002)
    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    xy1, xy2 = np.meshgrid(x, y)
    for blob in blobs_doh:
        distance = np.sqrt(
            np.sum(np.power(np.stack([xy1.flatten(), xy2.flatten()], axis=-1) - blob[:-1].reshape(1, -1), 2.0),
                   axis=-1))
        x_blob = xy1.flatten()[distance < blob[-1]].astype('uint16')
        y_blob = xy2.flatten()[distance < blob[-1]].astype('uint16')
        rain_mm = image[x_blob, y_blob].mean()
        print(rain_mm, len(x_blob), blob)

    fig, axes = plt.subplots(1, 3)
    ax = axes.ravel()
    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()
    plt.tight_layout()
    plt.show()
    # plt.imshow(image)
    # plt.show()
    # if i > 0:
    #     break
    # print("a")
