# %%
import gin
import torch
import pytorch_lightning

# %%
# Activation Functions
gin.config.external_configurable(torch.nn.ReLU, module='torch.nn')
gin.config.external_configurable(torch.nn.SiLU, module='torch.nn')
gin.config.external_configurable(torch.nn.Mish, module='torch.nn')

# Callbacks
gin.config.external_configurable(pytorch_lightning.callbacks.EarlyStopping, module='pytorch_lightning.callbacks')

# loggers
gin.config.external_configurable(pytorch_lightning.loggers.TensorBoardLogger, module='pytorch_lightning.loggers')
gin.config.external_configurable(pytorch_lightning.loggers.WandbLogger, module='pytorch_lightning.loggers')
