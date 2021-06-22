import torch
import torch.autograd as autograd
from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm
import copy
from gan.config import GANConfig

GENERATOR = 'Generator'
DISCRIMINATOR = 'Discriminator'


def get_value(t):
    if isinstance(t, torch.Tensor):
        return t.item()
    return t


# Make all layers to be spectral normalization layer
def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        return spectral_norm(m)
    else:
        return m


class BaseTrainer(object):
    def __init__(self, gan_config, steps: list):
        self.steps = steps
        self.gan_config = gan_config

    def get_steps(self) -> list:
        return self.steps

    def train_step(self, index, **kwargs):
        raise NotImplemented

    def samples_noise(self, batch_size=None) -> torch.Tensor:
        return torch.randn(self.gan_config.batch_size if batch_size is None else batch_size, self.gan_config.z_size,
                           requires_grad=True).to(
            self.gan_config.working_device)


class GANTraining(BaseTrainer):
    def __init__(self, gan_config: GANConfig, net_discriminator, net_g, input_optimizer_d, input_optimizer_g,
                 net_encoder=None):
        super(GANTraining, self).__init__(gan_config, [DISCRIMINATOR, GENERATOR])
        self.net_discriminator = net_discriminator
        self.net_g = net_g
        self.net_encoder = net_encoder
        self.has_encoder = net_encoder is not None
        if gan_config.is_spectral_norm():
            self.net_discriminator.apply(add_sn)
            print("Applying Spectral-Norm to discriminator")
        if gan_config.is_spectral_norm_generator():
            self.net_g.apply(add_sn)
            print("Applying Spectral-Norm to Generator")
        self.optimizer_d = input_optimizer_d
        self.optimizer_g = input_optimizer_g
        self.loss = gan_config.get_loss()
        self.net_g_best = None
        self.update_best()

    def train_step(self, step, **kwargs):
        if step == DISCRIMINATOR:
            return self.train_discriminator(kwargs['data'], condition=kwargs.get('condition'))
        elif step == GENERATOR:
            return self.train_generator(kwargs['data'], condition=kwargs.get('condition'))

    def calc_gradient_penalty(self, real_data, fake_data, condition):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1).to(self.gan_config.working_device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.to(self.gan_config.working_device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.net_discriminator(interpolates, condition)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(
                                      self.gan_config.working_device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_best(self):
        print("Updating Best Generator")
        del self.net_g_best
        self.net_g_best = copy.deepcopy(self.net_g)
        return self.net_g_best

    def get_generator_func(self):
        def func(batch_size=None, is_best=False, cond=None):
            z = self.samples_noise(batch_size=batch_size)
            if cond is not None and len(cond.shape) == 1:
                cond = cond.reshape([1, -1]).repeat([batch_size, 1])
            if is_best:
                if cond is None:
                    return self.net_g_best(z), z
                else:
                    return self.net_g_best(z, cond), z
            else:
                if cond is None:
                    return self.net_g(z), z
                else:
                    return self.net_g(z, cond), z

        return func

    def run_discriminator(self, data, condition):
        if self.gan_config.conditional:
            return self.net_discriminator(data, condition)
        else:
            return self.net_discriminator(data)

    def run_generator(self, data, condition, noise_vector=None):
        noise_vector = self.samples_noise(batch_size=data.shape[0]) if noise_vector is None else noise_vector
        if self.gan_config.conditional:
            return self.net_g(noise_vector, condition), noise_vector
        else:
            return self.net_g(noise_vector), noise_vector

    def train_discriminator(self, real_data, condition):
        gradient_penalty = 0
        for p in self.net_discriminator.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        self.net_discriminator.zero_grad()
        ########################################################
        # train with real
        ########################################################
        d_real = self.run_discriminator(real_data, condition)
        ########################################################
        # train with fake
        ########################################################
        gen_data, z = self.run_generator(real_data, condition)
        d_fake = self.run_discriminator(gen_data, condition)

        loss = self.loss.loss_critic(d_real, d_fake)
        loss.backward()
        ########################################################
        # Gradient Penalty
        ########################################################
        if self.gan_config.enable_gp() > 0:
            gradient_penalty = self.gan_config.gp_lambda * self.calc_gradient_penalty(real_data.data, gen_data.data,
                                                                                      condition)
            gradient_penalty.backward()
        ########################################################
        # Gradient Clipping
        ########################################################
        if self.gan_config.is_clipping():
            nn.utils.clip_grad_norm_(self.net_discriminator.parameters(), self.gan_config.clipping_value)
        d_cost = loss + gradient_penalty
        self.optimizer_d.step()
        return {'total_loss': get_value(d_cost),
                'gradient_penalty': get_value(gradient_penalty),
                'loss': get_value(loss)}

    def train_generator(self, real_data, condition):

        for p in self.net_discriminator.parameters():
            p.requires_grad = False  # to avoid computation
        self.net_g.zero_grad()

        gen_data, z = self.run_generator(real_data, condition)
        d_fake = self.run_discriminator(gen_data, condition)

        d_real = self.run_discriminator(real_data, condition) if self.loss.dual_generator else None
        loss_gen = self.loss.loss_generator(d_real, d_fake)
        #######################################################
        # VAE Loss
        ######################################################
        mse_loss = 0
        kl_loss = 0
        if self.has_encoder:
            mu, log_var = self.net_encoder(real_data, condition)
            z_hat = self.reparameterization(mu, log_var)
            fake_vae, _ = self.run_generator(real_data, condition, noise_vector=z_hat)
            mse_loss = torch.pow(fake_vae - real_data, 2.0).mean()  # MSE Loss
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = loss_gen + mse_loss + self.gan_config.kl_loss_factor * kl_loss
        loss_dict = {'total_loss': get_value(loss),
                     'loss_gen': get_value(loss_gen),
                     'kl_loss': get_value(kl_loss),
                     'mse_loss': get_value(mse_loss)}

        loss.backward()
        if self.gan_config.is_clipping():
            if self.has_encoder:
                nn.utils.clip_grad_norm_(self.net_encoder.parameters(), self.gan_config.clipping_value)
            nn.utils.clip_grad_norm_(self.net_g.parameters(), self.gan_config.clipping_value)
        self.optimizer_g.step()

        return loss_dict

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * self.samples_noise(batch_size=std.shape[0])
