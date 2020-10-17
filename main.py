from dataset.radar_static import RadarDataSet
from google.colab import drive

drive.mount('/content/gdrive/')

PROJECT = 'RainMapGenerator'

if __name__ == '__main__':
    print(f"Starting Run of {PROJECT}")
    data_file = '/content/gdrive/My Drive/Runners/Data/rain_data.pickle'
    rds = RadarDataSet(data_file)
    print(len(rds))
