# %%
import sys

import gin
import torch
import torch.nn as nn

# %%
@gin.configurable(denylist=['in_chan'])
class ResConv1DBlock(nn.Module):
    def __init__(self, in_chan, filters=128, kernel_size=3, dropout=0.25, activation=nn.ReLU(), dilation=1, residual=True):
        super(ResConv1DBlock, self).__init__()

        self.conv1d = nn.Conv1d(in_chan, filters, kernel_size=kernel_size, dilation=int(dilation), padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = activation
        self.dropout = nn.Dropout1d(dropout) if dropout is not None else None
        self.residual = residual

        self.linear = None
        if residual and (in_chan != filters):
            self.linear = nn.Conv1d(in_chan, filters, kernel_size=1, bias=False)

    @property
    def out_channels(self):
        return self.conv1d.out_channels
    
    def forward(self, inputs, **kwargs):
        x = inputs

        try:
            x = self.conv1d(x)
        except:
            print(x.shape, x.dtype, file=sys.stderr)
            raise

        x = self.batch_norm(x)
        x = self.act(x)
        # dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # residual
        if self.residual:
            if self.linear:
                inputs = self.linear(inputs)
            x = inputs + x

        return x

# %%
# @gin.configurable(denylist=['in_chan'])
# class BasenjiConv1DBlock(nn.Module):
#     def __init__(self, in_chan, filters=128, kernel_size=3, dropout=0.25, activation=nn.ReLU(), dilation=1, residual=True, pointwise_bias=False):
#         super(BasenjiConv1DBlock, self).__init__()

#         self.conv1d = nn.Conv1d(in_chan, filters, kernel_size, dilation=dilation, padding='same')
#         self.batch_norm = nn.BatchNorm1d(filters)
#         self.act = activation
#         self.pointwise = nn.Conv1d(filters, filters*2, kernel_size=1, bias=pointwise_bias)
#         self.dropout = nn.Dropout1d(dropout) if dropout is not None else None
#         self.residual = residual

#         self.linear_upsample = None
#         if residual and (in_chan != filters):
#             self.linear_upsample = nn.Conv1d(in_chan, filters, kernel_size=1, bias=False)

#         self.out_channels = filters*2
    
#     def forward(self, inputs, **kwargs):
#         x = inputs
#         # conv with C channels
#         x = self.conv1d(x)

#         # normalize and apply activation
#         x = self.batch_norm(x)
#         x = self.act(x)

#         # upsample to 2*C
#         x = self.pointwise(x)

#         # dropout
#         if self.dropout is not None:
#             x = self.dropout(x)

#         # residual
#         if self.residual:
#             if self.linear_upsample:
#                 inputs = self.linear_upsample(inputs)
#             x = inputs + x

#         return x

