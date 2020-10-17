import torch
from torchvision.transforms import RandomHorizontalFlip, \
    RandomVerticalFlip

from torchvision import transforms
from torch import optim
from tqdm import tqdm

import gan
from networks.factory import get_network
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization
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
lr = 1e-4
betas = (0.5, 0.999)
wd = 1e-4
epoch = 10
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

    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=betas, weight_decay=wd)
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=betas, weight_decay=wd)

    gan_cfg = gan.GANConfig(gan.GANType.RaSGAN, batch_size=batch_size, z_size=z_size,
                            input_working_device=working_device)
    gan_trainer = gan.GANTraining(gan_cfg, net_d, net_g, optimizer_d, optimizer_g)

    for i in range(epoch):
        for data in tqdm(train_loader):
            for step in gan_trainer.get_steps():
                gan_trainer.train_step(step, data=data)
