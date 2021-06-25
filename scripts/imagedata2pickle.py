import os
import pickle
import numpy as np
from tqdm import tqdm
from dataset.data_common import preprocess_radar_data, load_radar_image, slice
from matplotlib import pyplot as plt


def save_image2pickle(h, w, image_folder_path, n_samples, rain_t=0.1, rain_pixel_t=0.1, debug_plot=True):
    file_list = os.listdir(image_folder_path)
    print(len(file_list))

    index_list = np.linspace(0, len(file_list) - 1, len(file_list)).astype('uint64')
    np.random.shuffle(index_list)  # Select file randomly
    image_list = []
    filtered = 0
    for file_index in tqdm(index_list):
        file_path = os.path.join(image_folder_path, file_list[file_index])
        data = load_radar_image(file_path)
        rain_image, row, col = preprocess_radar_data(data, h, h, 5)
        if len(row) > 0:
            if debug_plot:
                plt.imshow(rain_image)
                plt.colorbar()
                plt.show()
            for r, c in zip(row, col):
                d = slice(data, r, c, h, w)
                rain_image_slice = slice(rain_image, r, c, h, w)
                dn = rain_image_slice / np.max(rain_image_slice)
                rain_pixels = np.sum(dn > rain_t) / (h * w)
                if rain_pixels > rain_pixel_t:
                    image_list.append(d)
                else:
                    filtered += 1
        if len(image_list) > n_samples:
            print("Finised Step One")
            break
        print(len(image_list) / n_samples, filtered)
    return image_list


if __name__ == '__main__':
    H = 32
    W = 32
    n_s = 10000
    data_folder_path = '/data/datasets/radar_maps'

    labled = False

    out_list = save_image2pickle(H, W, data_folder_path, 10000, debug_plot=True)
    pickle.dump(out_list, open('/data/datasets/rain_data_val.pickle', 'wb'))

    out_list = save_image2pickle(H, W, data_folder_path, 200000, debug_plot=False)
    pickle.dump(out_list, open('/data/datasets/rain_data.pickle', 'wb'))
