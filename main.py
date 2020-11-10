import torch
import argparse
from torchvision.transforms import RandomHorizontalFlip, \
    RandomVerticalFlip
from importlib import util
from torchvision import transforms
from torch import optim
from tqdm import tqdm

import gan
from networks.factory import get_network
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization
from dataset.blob_annotations import BlobAnnotations
from metric import ResultsAveraging, FrechetInceptionDistance
from matplotlib import pyplot as plt

wandb_flag = util.find_spec("wandb")
found_wandb = wandb_flag is not None
if found_wandb:
    print("Found ")
    import wandb

google_flag = util.find_spec("google")
google_flag = google_flag is not None
if google_flag:
    from google.colab import drive

PROJECT = 'RainMapGenerator'

# data_file = '/content/gdrive/My Drive/Runners/Data/rain_data.pickle'
# data_file_val = '/content/gdrive/My Drive/Runners/Data/rain_data_val.pickle'

batch_size = 32
h = 32
w = 32
dim = 128
# lr_g = 1e-4
# lr_d = 1e-4
betas = (0.5, 0.999)
wd = 1e-4


def arg_parsing():
    parser = argparse.ArgumentParser(description='Rain Map Generative Training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, default=50)

    parser.add_argument('--training_data_pickle', type=str,
                        default='/content/gdrive/My Drive/Runners/Data/rain_data.pickle')
    parser.add_argument('--validation_data_pickle', type=str,
                        default='/content/gdrive/My Drive/Runners/Data/rain_data_val.pickle')
    ################################
    # Optimizer
    ################################
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    ################################
    # GAN
    ################################
    parser.add_argument('--loss_type', type=str, default='WGAN', choices=['WGAN', 'RaSGAN'])
    parser.add_argument('--z_size', type=int, default=128)
    parser.add_argument('--sn_enable', action='store_true')
    parser.add_argument('--gp_lambda', type=float, default=10)
    ################################
    # VAE
    ################################
    parser.add_argument('--vae_enable', action='store_true')
    parser.add_argument('--lr_e', type=float, default=1e-4)
    parser.add_argument('--kl_loss_factor', type=float, default=3)
    ################################
    # Network Config
    ################################
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    if google_flag:
        print("Mounting Drive Folder...")
        drive.mount('/content/gdrive/')

    args = arg_parsing()
    print(f"Starting Run of {PROJECT}")
    if found_wandb:
        wandb.init(project=PROJECT)
        wandb.config.update(args)  # adds all of the arguments as config variables
    torch.manual_seed(args.seed)
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Working Device is set to:" + str(working_device))

    transform_training = transforms.Compose([
        transforms.ToTensor(),
        MaxNormalization(),
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-180, 180)),
        transforms.ToTensor(),
        BlobAnnotations(h, w)

    ])

    transform_validation = transforms.Compose([
        transforms.ToTensor(),
        MaxNormalization(),
    ])
    # TODO: add data augmentation
    train_rds = RadarDataSet(args.training_data_pickle, transform=transform_training)
    print(train_rds.data_shape)
    train_loader = torch.utils.data.DataLoader(dataset=train_rds,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_rds = RadarDataSet(args.validation_data_pickle, transform=transform_validation)

    validation_loader = torch.utils.data.DataLoader(dataset=val_rds,
                                                    batch_size=batch_size,
                                                    shuffle=False)
    fid = FrechetInceptionDistance(batch_size, validation_loader, working_device)
    net_g, net_d, net_e = get_network(args.z_size, dim, h, w, args.vae_enable, working_device)

    optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr_d, betas=betas, weight_decay=wd)
    optimizer_g = optim.Adam(net_g.parameters(), lr=args.lr_g, betas=betas, weight_decay=wd)
    optimizer_e = None
    if args.vae_enable and net_e is not None:
        optimizer_g = optim.Adam(net_e.parameters(), lr=args.lr_e, betas=betas, weight_decay=wd)

    gan_cfg = gan.GANConfig(gan.GANType[args.loss_type], batch_size=batch_size, z_size=args.z_size,
                            input_working_device=working_device, sn_enable=args.sn_enable, gp_lambda=args.gp_lambda,
                            kl_loss_factor=args.kl_loss_factor)
    gan_trainer = gan.GANTraining(gan_cfg, net_d, net_g, optimizer_d, optimizer_g, net_encoder=net_e,
                                  input_optimizer_e=optimizer_e)

    ra = ResultsAveraging()
    for i in range(args.n_epoch):
        for data, label in tqdm(train_loader):
            data = data.to(working_device)
            batch_results_dict = {}
            for step in gan_trainer.get_steps():
                loss_dict = gan_trainer.train_step(step, data=data)
                batch_results_dict.update({step + k: v for k, v in loss_dict.items()})
            ra.update_results(batch_results_dict)
        result_dict = ra.results()
        generative_func = gan_trainer.get_generator_func()
        fid_score = fid.calculate_fid(generative_func)
        if ra.is_best(fid_score):
            print("New Best :) everyone loves to play with GANs")
            gan_trainer.update_best()
            n_example = 4
            data, _ = generative_func(batch_size=n_example * n_example, is_best=True)
            data = data.detach().cpu().numpy().reshape(-1, h, w)
            for j in range(n_example * n_example):
                plt.subplot(n_example, n_example, j + 1)
                plt.imshow(data[j, :, :])

        result_dict.update({'FID': fid_score, 'examples': plt})
        wandb.log(result_dict)
        print(f"Finished Epoch:{i} with FID:{fid_score}")
