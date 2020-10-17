import torch
import torch.autograd as autograd
from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm
import copy
from gan.config import GANConfig


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
    def __init__(self, gan_config: GANConfig, net_discriminator, net_g, input_optimizer_d, input_optimizer_g):
        super(GANTraining, self).__init__(gan_config, ['Discriminator', 'Generator'])

        self.net_discriminator = net_discriminator
        self.net_g = net_g
        if gan_config.is_spectral_norm():
            self.net_discriminator.apply(add_sn)
            print("Applying Spectral-Norm to discriminator")
        self.optimizer_d = input_optimizer_d
        self.optimizer_g = input_optimizer_g
        self.loss = gan_config.get_loss()
        self.i_critic = 0
        self.net_g_best = None
        self.update_best()

    def train_step(self, step, **kwargs):
        if step == 'Discriminator':
            return self.train_discriminator(kwargs['data'], condition=kwargs.get('condition'))
        elif step == 'Generator':
            return self.train_generator(kwargs['data'], condition=kwargs.get('condition'))

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1).to(self.working_device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.to(self.working_device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.net_discriminator(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(
                                      self.working_device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_best(self):
        print("Updating Best Generator")
        del self.net_g_best
        self.net_g_best = copy.deepcopy(self.net_g)

    def get_generator_func(self):
        def func(batch_size=None, is_best=False, cond=None):
            z = self.samples_noise(batch_size=batch_size)
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
        self.i_critic += 1
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
        if self.gan_config.enable_gp() > 0:
            gradient_penalty = self.gan_config.gp_lambda * self.calc_gradient_penalty(real_data.data, gen_data.data)
            gradient_penalty.backward()

        d_cost = (loss + gradient_penalty).item()
        self.optimizer_d.step()
        return {'loss': d_cost}

    def train_generator(self, real_data, condition):

        for p in self.net_discriminator.parameters():
            p.requires_grad = False  # to avoid computation
        self.net_g.zero_grad()

        gen_data, z = self.run_generator(real_data, condition)
        d_fake = self.run_discriminator(gen_data, condition)

        d_real = self.run_discriminator(real_data, condition) if self.loss.dual_generator else None
        loss_gen = self.loss.loss_generator(d_real, d_fake)

        loss_dict = {'loss': loss_gen.item()}
        if self.i_critic % self.gan_config.n_critic == 0:
            loss = loss_gen

            loss_dict.update({'total_loss': loss.item()})
            loss.backward()
            self.optimizer_g.step()

        return loss_dict
