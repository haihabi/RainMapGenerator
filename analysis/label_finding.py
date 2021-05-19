from torchvision import transforms
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt

training_data_pickle = '/Users/haihabi/Downloads/rain_data.pickle'
transform_training = transforms.Compose([
    transforms.ToTensor(),
    MaxNormalization(),
])
# TODO: add data augmentation
train_rds = RadarDataSet(training_data_pickle, transform=transform_training)
for i in range(len(train_rds)):
    image = train_rds.get_sample(i)
    blobs_log = blob_doh(image, max_sigma=10, num_sigma=30, threshold=.001)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    # contours = measure.find_contours(image, 0.1, positive_orientation='high')
    # edges = feature.canny(image > 0.01, sigma=0.1)
    # Detect two radii
    # hough_radii = np.arange(1, 10, 1)
    # hough_res = hough_circle(edges, hough_radii)
    # accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
    #                                            total_num_peaks=20)
    # i = np.argsort(radii)
    # cx = cx[i][-10:]
    # cy = cy[i][-10:]
    # radii = radii[i][-10:]
    # plt.subplot(1, 2, 1)
    fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
    for blob in blobs_log:
        y, x, r = blob
        print(x,y,r)
        c = plt.Circle((x, y), r, linewidth=2, fill=False,color='red')
        ax.add_artist(c)
        # c.draw()
    # for center_y, center_x, radius in zip(cy, cx, radii):
    #     phase = np.linspace(-np.pi, np.pi, 50)
    #     x = center_x + radius * np.cos(phase)
    # y = center_y + radius * np.sin(phase)
    # print("a")
    # circy, circx = circle_perimeter(center_y, center_x, radius,
    #                                 shape=image.shape)
    # plt.plot(x, y)

    # for c in contours:
    #  plt.plot(c[:, 1], c[:, 0])
    # Expand numpy dimensions
    # c = np.expand_dims(c.astype(np.float32), 1)
    # Convert it to UMat object
    # c = cv2.UMat(c)
    # area = cv2.contourArea(c)
    # print(area)

    # print("a")

    plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(edges)
    plt.show()
    # if i > 0:
    #     break
    # print("a")
