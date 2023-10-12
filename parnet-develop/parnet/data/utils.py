# %%
import torch

from .datasets import TFIterableDataset, DummyIterableDataset

# %%
def tfrecord_to_dataloader(tfrecord, batch_size=128, shuffle=1_000_000):
    return torch.utils.data.DataLoader(TFIterableDataset(tfrecord, batch_size=batch_size, shuffle=shuffle), batch_size=None)

# %%
def dummy_dataloader(n):
    return torch.utils.data.DataLoader(DummyIterableDataset(n=n), batch_size=None)