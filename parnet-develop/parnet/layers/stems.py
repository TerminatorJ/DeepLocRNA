# %%
import gin
import torch
import torch.nn as nn

# %%
@gin.configurable()
class StemConv1D(nn.Module):
    def __init__(self, in_chan=4, filters=128, kernel_size=12, activation=nn.ReLU(), dropout=None):
        super(StemConv1D, self).__init__()

        self.conv1d = nn.Conv1d(in_chan, filters, kernel_size, padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = activation
        self.dropout = nn.Dropout1d(dropout) if dropout is not None else None

    @property
    def out_channels(self):
        return self.conv1d.out_channels
    
    def forward(self, x, **kwargs):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
