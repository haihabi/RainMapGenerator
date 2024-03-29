from enum import Enum
from gan.losses import LSGANLoss, WGANLoss, RaSGANLoss, RaLSGAN, GANLoss


class GANType(Enum):
    GAN = 1
    WGAN = 2
    LSGAN = 3
    RaSGAN = 6
    RaLSGAN = 8


def require(key, **kwargs):
    if kwargs.get(key) is None:
        raise Exception('Missing Key:' + key)


__loss_dict__ = {GANType.WGAN: WGANLoss(),
                 GANType.GAN: GANLoss(),
                 GANType.RaSGAN: RaSGANLoss(),
                 GANType.RaLSGAN: RaLSGAN(),
                 GANType.LSGAN: LSGANLoss()}


def key_exists(key, **kwargs):
    return kwargs.get(key) is not None


class GANConfig(object):
    def __init__(self, gan_type: GANType, batch_size: int, z_size: int, input_working_device, conditional: bool = False,
                 sn_enable=False, gp_lambda=0, kl_loss_factor=1, n_critic=1, clipping_value=0.1,
                 sn_enable_generator=False):
        self.gan_type = gan_type
        self.working_device = input_working_device
        self.batch_size = batch_size
        self.z_size = z_size
        self.conditional = conditional
        self.n_critic = 1
        self.gp_lambda = gp_lambda
        self.sn_enable = sn_enable
        self.sn_enable_generator = sn_enable_generator
        self.n_critic = n_critic
        self.kl_loss_factor = kl_loss_factor
        self.clipping_value = clipping_value

    def is_clipping(self) -> bool:
        return self.clipping_value is not None

    def is_spectral_norm(self):
        return self.sn_enable

    def is_spectral_norm_generator(self):
        return self.sn_enable_generator

    def enable_gp(self):
        return self.gp_lambda > 0

    def get_loss(self):
        loss = __loss_dict__.get(self.gan_type)
        if loss is None:
            raise Exception(f'cant find loss function for gan type:{self.gan_type.name}')
        return loss
