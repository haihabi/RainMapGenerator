import torch
from torch import nn
from networks.dcgan import Generator
from networks.twod_gaussian import TwoDGaussian


class RefinementGenerator(nn.Module):
    def __init__(self, dim, h, w, out_features=16, z_size=128):
        super(RefinementGenerator, self).__init__()
        self.dcgan = Generator(dim, h, w, out_features=out_features, z_size=z_size)
        self.two2gaussion = TwoDGaussian(h, w)

    def forward(self, input_tensor, cond):
        base_refinement = self.dcgan(input_tensor)
        gaussian_map = self.two2gaussion(cond)
        refinement_map = self.res_block1(base_refinement, gaussian_map)
        refinement_map = self.res_block2(base_refinement, refinement_map)
        return refinement_map
