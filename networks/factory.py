import torch
from enum import Enum
from networks.dcgan import DCGANGenerator, Discriminator, Encoder, ModifiedDCGANGenerator
from networks.dcgan_adain import AdaINDCGANGenerator
from huggingface_hub import hf_hub_url, cached_download
from common import get_working_device


class GeneratorType(Enum):
    DCGAN = 0
    MDCGAN = 1
    AdaINDCGAN = 2


def get_network(network_type, z_size, dim, h, w, vae_enable, condition_vector_size, input_working_device):
    if network_type == GeneratorType.DCGAN:
        net_g = DCGANGenerator(dim, h, w, z_size=z_size, condition_vector_size=condition_vector_size).to(
            input_working_device)
    elif network_type == GeneratorType.MDCGAN:
        net_g = ModifiedDCGANGenerator(dim, h, w, z_size=z_size, condition_vector_size=condition_vector_size).to(
            input_working_device)
    elif network_type == GeneratorType.AdaINDCGAN:
        net_g = AdaINDCGANGenerator(dim, h, w, z_size=z_size, condition_vector_size=condition_vector_size).to(
            input_working_device)
    else:
        raise NotImplemented
    net_d = Discriminator(dim, h, w, condition_vector_size=condition_vector_size).to(input_working_device)
    net_e = Encoder(dim, z_size, h, w, condition_vector_size=condition_vector_size).to(
        input_working_device) if vae_enable else None
    return net_g, net_d, net_e


def get_rain_gan_function(h, w):
    working_device = get_working_device()
    net_g = DCGANGenerator(128, h, w, z_size=128, condition_vector_size=2).to(
        working_device)
    hf_url = hf_hub_url("HVH/RainGAN", "RainGAN32x32.pt")
    file_path = cached_download(hf_url)
    net_g.load_state_dict(torch.load(file_path, map_location='cpu'))
    net_g.eval()

    def sample_rain_field(rain_coverage, n_peaks, peak_rain_rate, batch_size=1):
        with torch.no_grad():
            cond = torch.Tensor([rain_coverage, n_peaks]).reshape([1, -1]).repeat([batch_size, 1])
            z = torch.randn([batch_size, 128])
            return peak_rain_rate * net_g(z, cond).cpu().numpy()[:, 0, :, :]

    return sample_rain_field
