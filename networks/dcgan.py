import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, dim, h, w, out_features=16, z_size=128):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.h_in = int(h / 8)
        self.w_in = int(w / 8)
        self.h = h
        self.w = w

        self.out_features = out_features
        norm_class = nn.Identity
        preprocess = nn.Sequential(
            nn.Linear(self.z_size, 4 * self.h_in * self.w_in * dim),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            norm_class(),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2, padding=0),
            norm_class(),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(dim, self.out_features, 2, stride=2, padding=0)

        self.output_intensity = nn.Conv2d(self.out_features, 1, kernel_size=1, stride=1, padding=0)
        self.dim = dim
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.eps = 1e-6
        self.output_nl = nn.Sigmoid()

    def forward(self, input_tensor, cond=None):
        output = self.preprocess(input_tensor)
        output = output.view(-1, 4 * self.dim, self.h_in, self.w_in)
        output = self.block1(output)  # x2 8,8
        output = self.block2(output)  # x2 16,16
        output = self.deconv_out(output)
        output_intensity = self.output_nl(self.output_intensity(output))
        output = output_intensity
        return output.view(-1, 1, self.h, self.w)


class Discriminator(nn.Module):
    def __init__(self, dim, h, w):
        super(Discriminator, self).__init__()
        self.h = h
        self.w = w
        self.h_in = int(h / 8)
        self.w_in = int(w / 8)
        self.dim = dim
        main = nn.Sequential(
            nn.Conv2d(1, dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),

        )
        self.main = main
        self.output_size = (4 * self.h_in * self.w_in * dim)
        self.output = nn.Linear(self.output_size, 1)

    def forward(self, input, cond=None):
        input = input.view(-1, 1, self.h, self.w)
        out = self.main(input)
        out = out.view(-1, 4 * self.h_in * self.w_in * self.dim)
        out = self.output(out)
        return out.view(-1)


class Encoder(nn.Module):
    def __init__(self, dim, z_size, h, w):
        super(Encoder, self).__init__()
        self.h = h
        self.w = w
        self.z_size = z_size
        self.h_in = int(h / 8)
        self.w_in = int(w / 8)
        self.dim = dim
        main = nn.Sequential(
            nn.Conv2d(1, dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),

        )
        self.main = main
        self.output_size = (4 * self.h_in * self.w_in * dim)
        self.output_mu = nn.Linear(self.output_size, z_size)
        self.output_log_var = nn.Linear(self.output_size, z_size)

    def forward(self, input, cond=None):
        input = input.view(-1, 1, self.h, self.w)
        out = self.main(input)
        out = out.view(-1, 4 * self.h_in * self.w_in * self.dim)
        out_mu = self.output_mu(out)
        out_logvar = self.output_log_var(out)
        return out_mu.view(-1, self.z_size), out_logvar.view(-1, self.z_size)
