import torch
import torch.nn as nn
from src.models.custom_layers import PixelwiseNormalization, WSConv2d, UpSamplingBlock, WSLinear
from src.utils import get_transition_value
from src.models.base_model import ProgressiveBaseModel


def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
    return nn.Sequential(
        WSConv2d(in_dim, out_dim, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2),
        PixelwiseNormalization()
    )


class LatentReshape(nn.Module):

    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

    def forward(self, x):
        x = x.view(x.shape[0], self.latent_size, 4, 4)
        return x


class Generator(ProgressiveBaseModel):

    def __init__(self,
                 start_channel_dim,
                 image_channels,
                 latent_size):
        super().__init__(start_channel_dim, image_channels)
        # Transition blockss
        self.latent_size = latent_size
        self.to_rgb_new = WSConv2d(start_channel_dim, self.image_channels, 1, 0)
        self.to_rgb_old = WSConv2d(start_channel_dim, self.image_channels, 1, 0)

        self.core_blocks = nn.Sequential(
            nn.Sequential(
                WSLinear(self.latent_size, self.latent_size*4*4),
                LatentReshape(self.latent_size),
                conv_bn_relu(self.latent_size, self.latent_size, 3, 1)
            ))
        self.new_blocks = nn.Sequential()
        self.upsampling = UpSamplingBlock()

    def extend(self):
        output_dim = self.transition_channels[self.transition_step]
        # Downsampling module
        if self.transition_step == 0:
            core_blocks = nn.Sequential(
                *self.core_blocks.children(),
                UpSamplingBlock()
            )
            self.core_blocks = core_blocks
        else:
            self.core_blocks = nn.Sequential(
                *self.core_blocks.children(),
                self.new_blocks,
                UpSamplingBlock()
            )

        self.to_rgb_old = self.to_rgb_new
        self.to_rgb_new = WSConv2d(output_dim, self.image_channels, 1, 0)

        self.new_blocks = nn.Sequential(
            conv_bn_relu(self.prev_channel_extension, output_dim, 3, 1),
            conv_bn_relu(output_dim, output_dim, 3, 1),
        )
        super().extend()

    def new_parameters(self):
        new_paramters = list(self.new_blocks.parameters()) + list(self.to_rgb_new.parameters())
        return new_paramters

    def forward(self, z):
        x = self.core_blocks(z)
        if self.transition_step == 0:
            x = self.to_rgb_new(x)
            return x
        x_old = self.to_rgb_old(x)
        x_new = self.new_blocks(x)
        x_new = self.to_rgb_new(x_new)

        x = get_transition_value(x_old, x_new, self.transition_value)
        return x

    def device(self):
        return next(self.parameters()).device

    def generate_latent_variable(self, batch_size):
        return torch.randn(batch_size, self.latent_size, device=self.device())
