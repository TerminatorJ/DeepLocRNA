# %%
import gin
import torch
import torch.nn as nn

from .stems import StemConv1D
from .blocks import ResConv1DBlock

# %%
@gin.configurable()
class Conv1DTower(nn.Module):
    def __init__(self, in_channels, layers_filters=[(8, 256)], block_layer=ResConv1DBlock, dilation=1.0) -> None:
        super(Conv1DTower, self).__init__()





        # tower (of ResBock)
        layer_idx = 0
        self.tower = []
        prev_out_channels = in_channels
        if layers_filters is not None:
            for n, filters in layers_filters:
                blocks = []
                for _ in range(n):
                    block = block_layer(prev_out_channels, filters=filters, dilation=dilation**layer_idx)
                    layer_idx += 1 # after, because first round is dilation^0
                    blocks.append(block)
                    prev_out_channels = block.out_channels
                self.tower.append(nn.Sequential(*blocks))
        self.tower = nn.Sequential(*self.tower) # to nn.Module
    
    @property
    def out_channels(self):
        return self.tower[-1][-1].out_channels
    
    def forward(self, x):
        x = self.tower(x)
        return x
