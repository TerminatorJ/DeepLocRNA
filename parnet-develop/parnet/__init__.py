# %%
__version__ = '0.1.0'

# Disable tensorflow INFO and WARNING log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # This needs to be done *before* tensorflow is imported. 

# %%
# Disable absl INFO and WARNING log messages
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)


# %%
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# %%
# Import torch and additional external configurables 
import gin.torch.external_configurables
from . import gin_external_configurables

# %%
# Import parnet modules
from . import layers, networks, losses, metrics, data

# %%
import torch
# add tensor cores support
#torch.set_float32_matmul_precision('high')
