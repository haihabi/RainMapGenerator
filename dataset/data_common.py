import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_radar_files(save_path: str, file_type: str = 'tif') -> dict:
    def _file_name2datetime(file_name: str) -> pd.datetime:
        time_str = str.split(str.split(os.path.basename(file_name), '_')[-1], '.')[0]
        year = '20' + time_str[:2]
        day = time_str[4:6]
        month = time_str[2:4]
        hours = time_str[6:8]
        min = time_str[8:10]
        return pd.datetime(year=int(year), month=int(month), day=int(day), hour=int(hours), minute=int(min))

    return {_file_name2datetime(f): f for f in glob.glob(os.path.join(save_path, "*." + file_type))}


def load_radar_image(file_path):
    I = plt.imread(file_path)
    data = np.flipud(I)
    del I
    return data


def cumsum_diff_shift_horizontal(x, size, min_points=0):
    state_l2r = np.cumsum(x, axis=1) > min_points
    return state_l2r - np.concatenate([np.zeros((state_l2r.shape[0], size)), state_l2r[:, :-size]], axis=1)


def cumsum_diff_shift_vertical(x, size, min_points=0):
    state_l2r = np.cumsum(x, axis=0) > min_points
    return state_l2r - np.concatenate([np.zeros((size, state_l2r.shape[1])), state_l2r[:-size, :]], axis=0)


def in_range_map(data_image: np.ndarray, h, w):
    state_matrix = (data_image != 255).astype('int8') - (data_image == 255).astype('int8')
    state_l2r = np.cumsum(state_matrix, axis=1) > w
    state_r2l = np.fliplr(np.cumsum(np.fliplr(state_matrix), axis=1) > w)
    state_u2d = np.cumsum(state_matrix, axis=0) > h
    state_d2u = np.flipud(np.cumsum(np.flipud(state_matrix), axis=0) > h)
    # rows, cols = np.where(state_l2r * state_d2u * state_r2l * state_u2d)
    return state_l2r * state_d2u * state_r2l * state_u2d


def location_points(data_image: np.ndarray, h, w, range_map, min_points=0):
    state_matrix = (data_image != 0).astype('int8')

    state_l2r = cumsum_diff_shift_horizontal(state_matrix, h, min_points=min_points)
    state_r2l = np.fliplr(cumsum_diff_shift_horizontal(np.fliplr(state_matrix), h, min_points=min_points))
    state_u2d = cumsum_diff_shift_vertical(state_matrix, w, min_points=min_points)
    state_d2u = np.fliplr(cumsum_diff_shift_vertical(np.fliplr(state_matrix), w, min_points=min_points))

    rows, cols = np.where(state_l2r * state_d2u * state_r2l * state_u2d * range_map)
    return rows, cols


def preprocess_radar_data(data_image: np.ndarray, h, w, min_points):
    range_map = in_range_map(data_image, h, w)
    data_image = radar2rain(data_image)
    rows, cols = location_points(data_image, h, w, range_map, min_points=min_points)
    return data_image, rows, cols


def sample_slice(data_image: np.ndarray, r: np.ndarray, c: np.ndarray, h: int, w: int):
    index = int(np.random.randint(0, len(r) - 1, 1).astype('int'))
    di = data_image[r[index]:(r[index] + h), c[index]:(c[index] + w)]
    return di


def radar2rain(radar_tensor):
    gain = 0.4
    offset = -30
    dbz_tensor = radar_tensor * gain + offset
    radar_rain_tensor = np.power(10, ((dbz_tensor / 10) - np.log10(200)) * (1 / 1.5))
    radar_rain_tensor[dbz_tensor < 5] = 0
    radar_rain_tensor[radar_tensor == 255] = 0
    return radar_rain_tensor
