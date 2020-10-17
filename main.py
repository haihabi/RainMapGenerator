import torch
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization
from torchvision.transforms import RandomHorizontalFlip, \
    RandomVerticalFlip
from torchvision import transforms

from google.colab import drive

drive.mount('/content/gdrive/')
PROJECT = 'RainMapGenerator'

data_file = '/content/gdrive/My Drive/Runners/Data/rain_data.pickle'
seed = 0
batch_size = 32
if __name__ == '__main__':
    print(f"Starting Run of {PROJECT}")

    torch.manual_seed(seed)
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Working Device is set to:" + str(working_device))

    transform_training = transforms.Compose([
        transforms.ToTensor(),
        MaxNormalization(),
    ])
    train_rds = RadarDataSet(data_file, transform=transform_training)
    train_loader = torch.utils.data.DataLoader(dataset=train_rds,
                                               batch_size=batch_size,
                                               shuffle=True)
    for i in train_loader:
        print(i.shape)
