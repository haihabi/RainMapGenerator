import torch
from torch import nn


class TwoDGaussian(nn.Module):
    def __init__(self, h: int, w: int, x_min, x_max, y_min, y_max):
        super(TwoDGaussian, self).__init__()
        x_vector = torch.linspace(x_min, x_max, w)
        y_vector = torch.linspace(y_min, y_max, h)
        x_array, y_array = torch.meshgrid([x_vector, y_vector])
        self.x_array = nn.Parameter(x_array, requires_grad=False)
        self.y_array = nn.Parameter(y_array, requires_grad=False)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def single_random_gaussian_config(self, min_cells, max_cells):
        n_cells = torch.randint(min_cells, max_cells)
        mu_x = self.x_min + (self.x_max - self.x_min) * torch.rand(n_cells)
        mu_y = self.y_min + (self.y_max - self.y_min) * torch.rand(n_cells)

    def gaussian2d(self, r_c, mu_x, mu_y, sigma_x):
        exp_factor_x = torch.pow((self.x_array - mu_x), 2.0)
        exp_factor_y = torch.pow((self.y_array - mu_y), 2.0)
        exp_factor_scale = 1 / 2
        exp_factor = exp_factor_scale * (exp_factor_x + exp_factor_y) / sigma_x
        return r_c * torch.exp(-exp_factor)

    def forward(self, input_gaussian_list):
        res_list = []
        for batch_cfg in input_gaussian_list:  # Loop over batch
            output = None
            for r_c, mu_x, mu_y, sigma in batch_cfg:  # loop over cells
                if output is None:
                    output = self.gaussian2d(r_c, mu_x, mu_y, sigma)
                else:
                    output = output + self.gaussian2d(r_c, mu_x, mu_y, sigma)
            res_list.append(output)
        return torch.stack(res_list, dim=0)
