import torch
import argparse
import os
from importlib import util
from torchvision import transforms
from torch import optim
from tqdm import tqdm
from matplotlib import pyplot as plt

from networks.factory import get_network, GeneratorType
from dataset.radar_static import RadarDataSet
from dataset.preprocess import MaxNormalization, RadarImageAnnotation
from metric import ResultsAveraging, FrechetInceptionDistance
import gan

from common import get_working_device, PROJECT, init_folder, datetime_folder_name, M_CONDIONS

wandb_flag = util.find_spec("wandb")
found_wandb = wandb_flag is not None
if found_wandb:
    print("Found ")
    import wandb

google_flag = util.find_spec("google")
google_flag = google_flag is not None
if google_flag:
    try:
        from google.colab import drive
    except:
        google_flag = False


def arg_parsing():
    parser = argparse.ArgumentParser(description='Rain Map Generative Training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--training_data_pickle', type=str,
                        default='/content/data/rain_data.pickle' if google_flag else '/data/datasets/rain_data.pickle')
    parser.add_argument('--validation_data_pickle', type=str,
                        default='/content/data/rain_data_val.pickle' if google_flag else '/data/datasets/rain_data_val.pickle')

    parser.add_argument('--wandb_disable', action='store_false')
    parser.add_argument('--log_folder', type=str, default='./')
    ################################
    # Network Config
    ################################
    parser.add_argument('--h', type=int, default=32)
    parser.add_argument('--w', type=int, default=32)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--disable_conditional', action='store_true')
    ################################
    # Optimizer
    ################################
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    ################################
    # GAN
    ################################
    parser.add_argument('--generator_type', type=str, default='DCGAN', choices=[e.name for e in GeneratorType])
    parser.add_argument('--loss_type', type=str, default='RaSGAN', choices=['WGAN', 'RaSGAN'])
    parser.add_argument('--z_size', type=int, default=128)
    parser.add_argument('--sn_enable', action='store_true')
    parser.add_argument('--sn_enable_generator', action='store_true')
    parser.add_argument('--gp_lambda', type=float, default=-1)
    ################################
    # VAE
    ################################
    parser.add_argument('--vae_enable', action='store_true')
    parser.add_argument('--lr_e', type=float, default=1e-4)
    parser.add_argument('--kl_loss_factor', type=float, default=1)
    ################################
    # Network Config
    ################################
    args = parser.parse_args()
    return args


def mount_drive():
    if google_flag:
        print("Mounting Drive Folder...")
        drive.mount('/content/gdrive/')


def init_wandb(args):
    if found_wandb:
        wandb.init(project=PROJECT)
        wandb.config.update(args)  # adds all of the arguments as config variables


if __name__ == '__main__':
    print(f"Starting Run of {PROJECT}")
    mount_drive()
    args = arg_parsing()
    wandb_run_time_flag = args.wandb_disable and found_wandb
    if wandb_run_time_flag:
        init_wandb(args)
    else:
        log_folder = init_folder(args)

    torch.manual_seed(args.seed)
    working_device = get_working_device()

    transform_training_list = [
        transforms.ToTensor(),
        MaxNormalization(),
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ]

    transform_validation_list = [
        transforms.ToTensor(),
        MaxNormalization(),
    ]
    conditional = not args.disable_conditional
    if conditional:
        kernel_size = 13
        transform_training_list.append(RadarImageAnnotation(args.h, args.w, kernel_size, 0.1))
        transform_validation_list.append(RadarImageAnnotation(args.h, args.w, kernel_size, 0.1))

    transform_training = transforms.Compose(transform_training_list)
    transform_validation = transforms.Compose(transform_validation_list)

    train_rds = RadarDataSet(args.training_data_pickle, transform=transform_training)
    train_loader = torch.utils.data.DataLoader(dataset=train_rds,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    val_rds = RadarDataSet(args.validation_data_pickle, transform=transform_validation)

    validation_loader = torch.utils.data.DataLoader(dataset=val_rds,
                                                    batch_size=args.batch_size,
                                                    shuffle=False)
    print("Init Frechet Inception Distance")
    fid = FrechetInceptionDistance(args.batch_size, validation_loader, working_device, conditional=conditional)
    print("Init Networks")
    net_g, net_d, net_e = get_network(GeneratorType[args.generator_type], args.z_size, args.dim, args.h, args.w,
                                      args.vae_enable,
                                      M_CONDIONS if conditional else 0, working_device)
    print("Optimizers")
    betas = (args.beta1, args.beta2)
    optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr_d, betas=betas, weight_decay=args.weight_decay)
    optimizer_g = optim.Adam(net_g.parameters(), lr=args.lr_g, betas=betas)
    if net_e is not None:
        optimizer_g = optim.Adam(
            [{'params': net_e.parameters(), 'lr': args.lr_e, 'betas': betas, 'weight_decay': args.weight_decay},
             {'params': net_g.parameters(), 'lr': args.lr_g, 'betas': betas}])
    print("Init GAN Configuration")
    gan_cfg = gan.GANConfig(gan.GANType[args.loss_type], batch_size=args.batch_size, z_size=args.z_size,
                            conditional=conditional,
                            input_working_device=working_device, sn_enable=args.sn_enable, gp_lambda=args.gp_lambda,
                            kl_loss_factor=args.kl_loss_factor,
                            sn_enable_generator=args.sn_enable_generator)
    gan_trainer = gan.GANTraining(gan_cfg, net_d, net_g, optimizer_d, optimizer_g, net_encoder=net_e)

    ra = ResultsAveraging()
    print("Starting Training Loopr")
    for i in range(args.n_epoch):
        for data in tqdm(train_loader):
            if conditional:
                image = data[0].to(working_device)
                label = data[1].to(working_device)
            else:
                image = data.to(working_device)
                label = None
            batch_results_dict = {}
            for step in gan_trainer.get_steps():
                loss_dict = gan_trainer.train_step(step, data=image, condition=label)

                batch_results_dict.update({step + k: v for k, v in loss_dict.items()})
            ra.update_results(batch_results_dict)
        result_dict = ra.results()
        generative_func = gan_trainer.get_generator_func()
        fid_score = fid.calculate_fid(generative_func)
        ra.end_epoch(extra_results={'FID': fid_score})
        if ra.is_best(fid_score):
            print("New Best :) everyone loves to play with GANs")
            net2save = gan_trainer.update_best()
            torch.save(net2save.state_dict(), os.path.join(wandb.run.dir, "model_best.pt"))

            n_example = 4
            data, _ = generative_func(batch_size=n_example * n_example, is_best=True,
                                      cond=label[0, :] if conditional else None)
            data = data.detach().cpu().numpy().reshape(-1, args.h, args.w)
            for j in range(n_example * n_example):
                plt.subplot(n_example, n_example, j + 1)
                plt.imshow(data[j, :, :])

        if wandb_run_time_flag:
            result_dict.update({'FID': fid_score, 'examples': plt})

            wandb.log(result_dict)
        else:
            pass

        print(f"Finished Epoch:{i} with FID:{fid_score}")
