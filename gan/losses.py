import torch


def relativistic_average(c_base, c_mean):
    return c_base - c_mean.mean(dim=0)


def calculate_d_tilde(c_real, c_face):
    return torch.sigmoid(c_real - c_face.mean(dim=0))


class BaseGANLoss(object):
    def __init__(self, dual_generator):
        self.dual_generator = dual_generator

    @staticmethod
    def loss_critic(d_real, d_fake):
        raise NotImplemented

    @staticmethod
    def loss_generator(d_real, d_fake):
        raise NotImplemented


class WGANLoss(BaseGANLoss):
    def __init__(self):
        super(WGANLoss).__init__(False)

    @staticmethod
    def loss_critic(d_real, d_fake):
        fake_loss = d_fake.mean()
        real_loss = -d_real.mean()
        return fake_loss + real_loss

    @staticmethod
    def loss_generator(d_real, d_fake):
        fake_loss = -d_fake.mean()
        return fake_loss


class GANLoss(BaseGANLoss):
    def __init__(self):
        super(GANLoss).__init__(False)

    @staticmethod
    def loss_critic(d_real, d_fake):
        d_fake = torch.sigmoid(d_fake)
        d_real = torch.sigmoid(d_real)
        loss = -(torch.log(d_real + 1e-6) + torch.log(1 - d_fake + 1e-6)).mean()
        return loss

    @staticmethod
    def loss_generator(d_real, d_fake):
        d_fake = torch.sigmoid(d_fake)
        loss = (torch.log(1 - d_fake + 1e-6)).mean()
        return loss


class LSGANLoss(BaseGANLoss):
    def __init__(self, a=0, b=1, c=1):
        super(LSGANLoss).__init__(False)
        self.a = a
        self.b = b
        self.c = c

    def loss_critic(self, d_real, d_fake):
        loss_real = torch.pow(d_real - 1, 2.0).mean()
        loss_fake = torch.pow(d_fake, 2.0).mean()
        return 0.5 * (loss_real + loss_fake)

    def loss_generator(self, d_real, d_fake):
        loss_fake = torch.pow(d_fake - 1, 2.0).mean()
        return 0.5 * (loss_fake)


class RaSGANLoss(BaseGANLoss):
    def __init__(self):
        super(RaSGANLoss).__init__(True)

    @staticmethod
    def loss_critic(d_real, d_fake):
        loss_real = -torch.log(calculate_d_tilde(d_real, d_fake) + 1e-6).mean()
        loss_fake = -torch.log(1e-6 + 1.0 - calculate_d_tilde(d_fake, d_real)).mean()
        return loss_real + loss_fake

    @staticmethod
    def loss_generator(d_real, d_fake):
        loss_real = -torch.log(calculate_d_tilde(d_fake, d_real) + 1e-6).mean()
        loss_fake = -torch.log(1e-6 + 1.0 - calculate_d_tilde(d_real, d_fake)).mean()
        return loss_real + loss_fake


class RaLSGAN(BaseGANLoss):
    def __init__(self):
        super(RaLSGAN).__init__(True)

    @staticmethod
    def loss_critic(d_real, d_fake):
        loss_real = torch.pow(relativistic_average(d_real, d_fake) - 1, 2.0).mean()
        loss_fake = torch.pow(relativistic_average(d_fake, d_real) + 1, 2.0).mean()

        return loss_real + loss_fake

    @staticmethod
    def loss_generator(d_real, d_fake):
        loss_real = torch.pow(relativistic_average(d_fake, d_real) - 1, 2.0).mean()
        loss_fake = torch.pow(relativistic_average(d_real, d_fake) + 1, 2.0).mean()
        return loss_real + loss_fake
