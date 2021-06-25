import torch
import numpy as np
from dataset.data_common import preprocess_radar_data, load_radar_image, slice
from matplotlib import pyplot as plt

file_name = '/data/datasets/radar_maps/radar_1702012205.tif'

data = load_radar_image(file_name)
h, w = 32, 32
rain_t = 0.1
rain_pixel_t = 0.1
rain_image, r, c = preprocess_radar_data(data, h, w, 5)
image_list_above = []
image_list_above_i = []
image_list_below = []
image_list_below_i = []
image_list_rain_pixels = []


def plot_box(r_i, c_i, in_h, in_w, color='red', label=None):
    plt.plot([c_i, c_i + w], [r_i, r_i], color, label=label)
    plt.plot([c_i, c_i], [r_i, r_i + h], color)
    plt.plot([c_i + w, c_i + w], [r_i, r_i + h], color)
    plt.plot([c_i, c_i + w], [r_i + h, r_i + h], color)


for ri, ci in zip(r, c):
    # d = slice(data, ri, ci, h, w)
    rain_image_slice = slice(rain_image, ri, ci, h, w)
    dn = rain_image_slice / np.max(rain_image_slice)
    rain_pixels = np.sum(dn > rain_t) / (h * w)
    if rain_pixels > rain_pixel_t:
        image_list_above.append(rain_image_slice)
        image_list_above_i.append([ri, ci])
    else:
        image_list_below.append(rain_image_slice)
        image_list_rain_pixels.append(rain_pixels)
        image_list_below_i.append([ri, ci])
print("a")

i = np.argmin(image_list_rain_pixels)
plt.subplot(1, 3, 1)
plt.imshow(data)
plot_box(image_list_above_i[0][0], image_list_above_i[0][1], h, w, 'red', label='Passed Crop')
plot_box(image_list_below_i[i][0], image_list_below_i[i][1], h, w, 'green', label='Filtered Crop')
y_min = np.minimum(image_list_above_i[0][0], image_list_below_i[i][0]) - 10
y_max = np.maximum(image_list_above_i[0][0], image_list_below_i[i][0]) + 10 + h
plt.ylim(y_min, y_max)

x_min = np.minimum(image_list_above_i[0][1], image_list_below_i[i][1]) - 10
x_max = np.maximum(image_list_above_i[0][1], image_list_below_i[i][1]) + 10 + w
plt.xlim(x_min, x_max)
plt.legend()
plt.title('Original Radar Map')
plt.subplot(2, 3, 2)
plt.imshow(np.flipud(image_list_above[0]))

plt.colorbar()
plt.title('Example Of Passed Crop')
plt.subplot(2, 3, 5)
plt.imshow(np.flipud(image_list_below[i]))
plt.title('Example Of Filtered Crop')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(np.flipud(image_list_above[0] / np.max(image_list_above[0])))
plt.colorbar()
plt.title('Normalized Rain Field')
plt.show()

####################
# Rain Pixels
####################
b = np.flipud(image_list_above[0] / np.max(image_list_above[0]))
plt.subplot(1, 2, 1)
plt.imshow(b)
plt.title('Normalized Rain Field')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(b > 0.1)
plt.title('Binary Rain Field')
plt.show()

####################
# Rain Peaks
####################
kernel = 17
bt = torch.from_numpy(b.copy())
mp = torch.nn.MaxPool2d(kernel_size=kernel, padding=int(kernel / 2), stride=1)
bt_max = mp(bt.unsqueeze(dim=0).unsqueeze(dim=0)).squeeze(dim=0).squeeze(dim=0)
eq = (bt_max == bt) * (bt_max != 0).detach().numpy()
t_max = bt_max.detach().numpy()

plt.subplot(1, 3, 1)
plt.imshow(b)
plt.title("Normalized Rain Field")
plt.subplot(1, 3, 2)
plt.imshow(t_max)
plt.title("Max Pool Output")
plt.subplot(1, 3, 3)
plt.imshow(eq)
plt.title("Peak Map")
plt.show()

print("a")
