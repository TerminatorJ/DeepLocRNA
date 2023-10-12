# %%
import torch
import torch.nn as nn

# %%
class IndexEmbeddingOutput(nn.Module):
    def __init__(self, dim, num_tasks):
        super(IndexEmbeddingOutput, self).__init__()
    
        self.conv1d = nn.Conv1d(dim, num_tasks, kernel_size=1, bias=True)
    
    @property
    def embedding(self):
        return torch.squeeze(self.conv1d.weight, dim=-1)
    
    @property
    def out_channels(self):
        return self.conv1d.out_channels

    def forward(self, x, **kwargs):
        return self.conv1d(x)
