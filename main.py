from google.colab import drive

drive.mount('/content/gdrive/')
import os
import pickle

PROJECT = 'RainMapGenerator'

if __name__ == '__main__':
    print(f"Starting Run of {PROJECT}")
    data_file = '/content/gdrive/My Drive/Runners/Data/rain_data.pickle'
    data = pickle.load(open(data_file, 'rb'))
    print(len(data))
