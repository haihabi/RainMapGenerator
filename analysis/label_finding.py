from torchvision import transforms
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
import cv2

training_data_pickle = '/data/datasets/rain_data.pickle'
transform_training = transforms.Compose([
    transforms.ToTensor(),
    MaxNormalization(),
])
# TODO: add data augmentation
train_rds = RadarDataSet(training_data_pickle, transform=transform_training)
for i in range(len(train_rds)):
    image = train_rds.get_sample(i)
    contours = measure.find_contours(image, 0.1, positive_orientation='high')

    print()
    for c in contours:
        plt.plot(c[:, 1], c[:, 0])
        # Expand numpy dimensions
        c = np.expand_dims(c.astype(np.float32), 1)
        # Convert it to UMat object
        c = cv2.UMat(c)
        area = cv2.contourArea(c)
        print(area)

        # print("a")
    plt.imshow(image)
    plt.show()
    if i > 0:
        break
    # print("a")
