import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, dim, conditional, z_size=128, condition_size=2):
        super(Generator, self).__init__()
        self.conditional = conditional
        self.z_size = z_size
        self.out_features = 16
        if conditional:
            self.z_size += condition_size
        norm_class = nn.Identity
        preprocess = nn.Sequential(
            nn.Linear(self.z_size, 4 * 4 * 4 * dim),
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

    def forward(self, input, cond=None):
        if self.conditional:
            mu = cond[:, 1]
            spot_max = torch.sqrt(mu - torch.pow(mu, 2.0)) / (mu + self.eps)
            spot = cond[:, 0]
            spot = torch.min(spot, spot_max)
            cond = torch.stack([spot, mu], dim=1)
            input = torch.cat([input, cond], dim=1)
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.block1(output)  # x2 8,8
        output = self.block2(output)  # x2 16,16
        output = self.deconv_out(output)

        output_intensity = self.output_nl(self.output_intensity(output))

        output = output_intensity
        return output.view(-1, 1, 32, 32)


class Discriminator(nn.Module):
    def __init__(self, dim, conditional, condition_size=2):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.conditional = conditional
        main = nn.Sequential(
            nn.Conv2d(1, dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),

            nn.LeakyReLU(inplace=True),

        )
        self.main = main
        self.output_size = (4 * 4 * 4 * dim)
        if self.conditional:
            self.output_size += condition_size
        self.output = nn.Linear(self.output_size, 1)

    def forward(self, input, cond=None):
        input = input.view(-1, 1, 32, 32)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * self.dim)
        if self.conditional:
            out = torch.cat([out, cond], dim=1)
        out = self.output(out)
        return out.view(-1)
