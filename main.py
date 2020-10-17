from google.colab import drive

drive.mount('/content/gdrive')
import os

PROJECT = 'RainMapGenerator'

if __name__ == '__main__':
    print(f"Starting Run of {PROJECT}")
    data_folder = '/content/gdrive/My Drive/Runners/Data/rain_map_data/'
    print(len(os.listdir(data_folder)))
