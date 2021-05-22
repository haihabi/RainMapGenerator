from enum import Enum
from networks.dcgan import Generator, Discriminator, Encoder


class NetworkName(Enum):
    DCGAN = 0
    REFINEMENT = 1


def get_network(z_size, dim, h, w, vae_enable, condition_vector_size, input_working_device):
    net_g = Generator(dim, h, w, z_size=z_size, condition_vector_size=condition_vector_size).to(input_working_device)
    net_d = Discriminator(dim, h, w, condition_vector_size=condition_vector_size).to(input_working_device)
    net_e = Encoder(dim, z_size, h, w, condition_vector_size=condition_vector_size).to(
        input_working_device) if vae_enable else None
    return net_g, net_d, net_e
