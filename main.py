import os

PROJECT = 'RainMapGenerator'

if __name__ == '__main__':
    print(f"Starting Run of {PROJECT}")
    data_folder = '/content/gdrive/My Drive/Runners/rain_map_data/'
    print(len(os.listdir(data_folder)))
