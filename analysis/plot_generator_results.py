import matplotlib.pyplot as plt
import wandb
import argparse
from common import PROJECT, get_working_device, M_CONDIONS
import os
from networks.factory import get_network, GeneratorType
import torch


def argument():
    argp = argparse.ArgumentParser()
    argp.add_argument('--run_name', type=str)
    return argp.parse_args()


def load_model(cfg, weigths_path):
    working_device = get_working_device()
    conditional = not cfg.get('disable_conditional', False)
    net_g, _, _ = get_network(GeneratorType[cfg['generator_type']], cfg['z_size'], cfg.get('dim', 128),
                              cfg.get('h', 32), cfg.get('w', 32), cfg['vae_enable'],
                              M_CONDIONS if conditional else 0, working_device)
    net_g.load_state_dict(torch.load(weigths_path, map_location='cpu'))
    net_g.eval()

    def sample_rain_field(rain_coverage, n_peaks, batch_size=1):
        with torch.no_grad():
            cond = torch.Tensor([rain_coverage, n_peaks]).reshape([1, -1]).repeat([batch_size, 1])
            z = torch.randn([batch_size, cfg['z_size']])
            return net_g(z, cond).cpu().numpy()[:, 0, :, :]

    return net_g, sample_rain_field


def download_network_and_config(run_name):
    api = wandb.Api()
    runs = api.runs(f"hvh/{PROJECT}")
    run = None
    for r in runs:
        run = r if r.name == run_name else run
    cfg = run.config
    run.file("model_best.pt").download(replace=True)
    model_path = os.path.join(os.getcwd(), 'model_best.pt')
    net_g, sample_rain_field = load_model(cfg, model_path)
    return cfg, net_g, sample_rain_field


if __name__ == '__main__':
    cfg = argument()
    cfg, net_g, sample_rain_field = download_network_and_config(cfg.run_name)

    ###########################
    # Model Collapse Plot
    ###########################
    k = 4
    sample = sample_rain_field(0.1, 5, batch_size=k ** 2)
    for i in range(k):
        for j in range(k):
            plt.subplot(k, k, i + 1 + 4 * j)
            plt.imshow(sample[i + 4 * j, :, :])

    plt.show()
    ###########################
    # Rain Coverage Plot
    ###########################
    rain_coverage = [0.1, 0.4, 0.7, 0.9]
    for i, r in enumerate(rain_coverage):
        plt.subplot(1, len(rain_coverage), i + 1)
        sample = sample_rain_field(r, 5, batch_size=1)
        plt.imshow(sample[0, :, :])
        plt.title(f"$R_c={r}$")
    plt.show()
    ###########################
    # N Peaks Plot
    ###########################
    n_peaks_array = [5, 10, 20, 30]
    for i, n_p in enumerate(n_peaks_array):
        plt.subplot(1, len(n_peaks_array), i + 1)
        sample = sample_rain_field(0.3, n_p, batch_size=1)
        plt.imshow(sample[0, :, :])
        plt.title(f"$N_p={n_p}$")
    plt.show()
