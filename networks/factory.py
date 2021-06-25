from enum import Enum
from networks.dcgan import DCGANGenerator, Discriminator, Encoder, ModifiedDCGANGenerator
from networks.dcgan_adain import AdaINDCGANGenerator


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
