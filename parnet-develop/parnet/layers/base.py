# %%
import gin
import torch
import torch.nn as nn

# %%
@gin.configurable()
class LinearProjection(nn.Module):
    def __init__(self, in_features, out_features=128, activation=None) -> None:
        super(LinearProjection, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = activation
    
    def forward(self, x):
        x = self.linear(x)
        if self.act is not None:
            x = self.act(x)
        return x

# %%
@gin.configurable()
class Pointwise(nn.Module):
    def __init__(self, in_channels, out_channels=128, bias=False, activation=None) -> None:
        super(Pointwise, self).__init__()

        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.act = activation

    @property
    def out_channels(self):
        return self.pointwise.out_channels
    
    def forward(self, x):
        x = self.pointwise(x)
        if self.act is not None:
            x = self.act(x)
        return x
