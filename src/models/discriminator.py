import torch.nn as nn
from src.models.custom_layers import WSConv2d, WSLinear, MinibatchStdLayer
from src.utils import get_transition_value
from src.models.base_model import ProgressiveBaseModel


def conv_module_bn(dim_in, dim_out, kernel_size, padding):
    return nn.Sequential(
        WSConv2d(dim_in, dim_out, kernel_size, padding),
        nn.LeakyReLU(negative_slope=.2)
    )


class Discriminator(ProgressiveBaseModel):

    def __init__(self, 
                 image_channels,
                 start_channel_dim
                 ):
        super().__init__(start_channel_dim, image_channels)
        
        self.from_rgb_new = conv_module_bn(image_channels, start_channel_dim, 1, 0)

        self.from_rgb_old = conv_module_bn(image_channels, start_channel_dim, 1, 0)
        self.new_block = nn.Sequential()
        self.core_model = nn.Sequential(
            nn.Sequential(
                MinibatchStdLayer(),
                conv_module_bn(start_channel_dim+1, start_channel_dim, 3, 1),
                conv_module_bn(start_channel_dim, start_channel_dim, 4, 0),
            )
        )
        self.output_layer = WSLinear(start_channel_dim, 1)

    def extend(self):
        input_dim = self.transition_channels[self.transition_step] 
        output_dim = self.prev_channel_extension
        if self.transition_step != 0:
            self.core_model = nn.Sequential(
                self.new_block,
                *self.core_model.children()
            )
        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2, 2]),
            self.from_rgb_new
        )
        self.from_rgb_new = conv_module_bn(self.image_channels, input_dim, 1, 0)
        self.new_block = nn.Sequential(
            conv_module_bn(input_dim, input_dim, 3, 1),
            conv_module_bn(input_dim, output_dim, 3, 1),
            nn.AvgPool2d([2, 2])
        )
        self.new_block = self.new_block
        super().extend()

    def forward(self, x):
        x_old = self.from_rgb_old(x)
        x_new = self.from_rgb_new(x)
        x_new = self.new_block(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        x = self.core_model(x)
        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        return x
