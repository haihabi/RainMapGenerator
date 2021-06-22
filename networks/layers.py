from torch import nn


class AdaIN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AdaIN, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.mu_linear = nn.Linear(input_channels, output_channels, bias=False)
        self.std_linear = nn.Linear(input_channels, output_channels, bias=False)
        self.norm = nn.LayerNorm(output_channels, elementwise_affine=False)

    def forward(self, x, y):
        mu = self.mu_linear(y).reshape([-1, self.output_channels, 1, 1])
        std = self.std_linear(y).reshape([-1, self.output_channels, 1, 1])
        return std * self.norm(x) + mu
