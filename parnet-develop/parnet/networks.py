import sys

import gin
import torch
import torch.nn as nn

# %%
from .layers import StemConv1D, Conv1DTower, Pointwise, IndexEmbeddingOutput

# %%
@gin.configurable()
class PanRBPNet(nn.Module):
    def __init__(self, num_tasks, dim=128):
        super(PanRBPNet, self).__init__()

        self.num_tasks = num_tasks
        self.stem = StemConv1D()
        self.body = Conv1DTower(self.stem.out_channels)
        self.pointwise = Pointwise(self.body.out_channels, dim)
        self.output = IndexEmbeddingOutput(dim, num_tasks)

    def forward(self, inputs, **kwargs):
        x = inputs
        x = self.body(self.stem(x))
        x = self.pointwise(x)
        # x.shape: (batch_size, dim, N)
        #print("changed")
        #try: 
        #    x = self.output(x)
            # x.shape: (batch_size, num_tasks, N)
        #except:
        #    print(x.shape, x.dtype)
        #    raise

        return x
