import torch
import numpy as np
from torch import nn


# https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
class WSConv2d(nn.Module):
    """
    This is the wt scaling conv layer layer. Initialize with N(0, scale). Then 
    it will multiply the scale for every forward pass
    """
    def __init__(self, inCh, outCh, kernelSize, padding, gain=np.sqrt(2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kernelSize, stride=1, padding=padding)

        # new bias to use after wscale
        bias = self.conv.bias
        self.bias = nn.Parameter(bias.view(1, bias.shape[0], 1, 1))
        self.conv.bias = None

        # calc wt scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:]) # Leave out # of op filters
        self.wtScale = gain/np.sqrt(fanIn)

        # init
        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)
        self.name = '(inp = %s)' % (self.conv.__class__.__name__ + str(convShape))

    def forward(self, x):
        #return self.conv(x)
        return self.conv(x * self.wtScale) + self.bias

    def __repr__(self):
        return self.__class__.__name__ + self.name


class WSLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bias = self.linear.bias
        self.linear.bias = None
        fanIn = in_dim
        self.wtScale = np.sqrt(2) / np.sqrt(fanIn)

        nn.init.normal_(self.linear.weight)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x):
        return self.linear(x * self.wtScale) + self.bias


class PixelwiseNormalization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        factor = ((x**2).mean(dim=1, keepdim=True) + 1e-8)**0.5
        return x / factor


class UpSamplingBlock(nn.Module):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


class MinibatchStdLayer(nn.Module):

    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    # Implementation from:
    # https://github.com/facebookresearch/pytorch_GAN_zoo/blob/master/models/networks/custom_layers.py
    def forward(self, x):
        size = x.size()
        subGroupSize = min(size[0], self.group_size)
        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]
        G = int(size[0] / subGroupSize)
        if subGroupSize > 1:
            y = x.view(-1, subGroupSize, size[1], size[2], size[3])
            y = torch.var(y, 1)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
    
        return torch.cat([x, y], dim=1)