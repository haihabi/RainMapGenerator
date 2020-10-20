from enum import Enum
from networks.dcgan import Generator, Discriminator


class NetworkName(Enum):
    DCGAN = 0


def get_network(z_size, dim, h, w, vae_enable, input_working_device):
    net_g = Generator(dim, h, w, z_size=z_size).to(input_working_device)
    net_d = Discriminator(dim, h, w).to(input_working_device)
    net_e = Discriminator() if vae_enable else None
    return net_g, net_d
