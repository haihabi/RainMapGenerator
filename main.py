import torch
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization
from torchvision.transforms import RandomHorizontalFlip, \
    RandomVerticalFlip
from networks.factory import get_network
from torchvision import transforms

from google.colab import drive

drive.mount('/content/gdrive/')
PROJECT = 'RainMapGenerator'

data_file = '/content/gdrive/My Drive/Runners/Data/rain_data.pickle'
seed = 0
batch_size = 32
h = 32
w = 32
z_size = 128
dim = 128
if __name__ == '__main__':
    print(f"Starting Run of {PROJECT}")

    torch.manual_seed(seed)
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Working Device is set to:" + str(working_device))

    transform_training = transforms.Compose([
        transforms.ToTensor(),
        MaxNormalization(),
    ])
    # TODO: add data augmentation
    train_rds = RadarDataSet(data_file, transform=transform_training)
    print(train_rds.data_shape)
    train_loader = torch.utils.data.DataLoader(dataset=train_rds,
                                               batch_size=batch_size,
                                               shuffle=True)

    net_g, net_d = get_network(z_size, dim, h, w, working_device)
