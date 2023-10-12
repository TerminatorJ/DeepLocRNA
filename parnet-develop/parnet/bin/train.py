# %%
import datetime
import shutil
from pathlib import Path

import gin
import gin.torch.external_configurables # TODO: gin.torch doesn't actually import gin.torch.external_configurables --> PR?
import click
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .. import layers
from ..networks import PanRBPNet
from ..losses import MultinomialNLLLossFromLogits
from ..data.datasets import TFIterableDataset
from ..data import tfrecord_to_dataloader, dummy_dataloader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


# %%
@gin.configurable()
class Model(pl.LightningModule):
    def __init__(self, network=PanRBPNet, _example_input=None, loss=MultinomialNLLLossFromLogits, metrics=None, optimizer=torch.optim.Adam):
        super().__init__()
        self.network = network

        # loss
        self.loss_fn = nn.ModuleDict({
            'TRAIN': loss(),
            'VAL': loss(),
        })
        
        # metrics
        if metrics is None:
            metrics = {}
        self.metrics = nn.ModuleDict({
            'TRAIN': nn.ModuleDict({name: metric() for name, metric in metrics.items()}),
            'VAL': nn.ModuleDict({name: metric() for name, metric in metrics.items()}),
        })
        
        # optimizer
        self.optimizer_cls = optimizer

        # save hyperparameters
        self.save_hyperparameters()
    
    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx=None, **kwargs):
        inputs, y = batch
        y = y['total']
        y_pred = self.forward(inputs)
        # log total counts
        self.log('log10-1p_total_counts', torch.log10(y.sum()+1), on_step=True, logger=True)

        loss = self.compute_and_log_loss(y, y_pred, partition='TRAIN')
        self.compute_and_log_metics(y, y_pred, partition='TRAIN')
        return loss

    def validation_step(self, batch, batch_idx=None, **kwargs):
        inputs, y = batch
        y = y['total']
        y_pred = self.forward(inputs)
        self.compute_and_log_loss(y, y_pred, partition='VAL')
        self.compute_and_log_metics(y, y_pred, partition='VAL')
    
    def compute_and_log_loss(self, y, y_pred, partition=None):
        # on_step = False
        # if partition == 'TRAIN':
        #     on_step = True

        loss = self.loss_fn[partition](y, y_pred)
        self.log(f'loss/{partition}', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def compute_and_log_metics(self, y, y_pred, partition=None):
        # on_step = False
        # if partition == 'TRAIN':
        #     on_step = True

        for name, metric in self.metrics[partition].items():
            metric(y, y_pred)
            self.log(f'{name}/{partition}', metric, on_step=True, on_epoch=True, prog_bar=False)

# %%
def _make_callbacks(output_path, with_validation=False):
    callbacks = [
        ModelCheckpoint(dirpath=output_path/'checkpoints', every_n_epochs=1, save_last=True, save_top_k=1),
        LearningRateMonitor('step', log_momentum=True),
    ]
    # if with_validation:
    #     callbacks.append(EarlyStopping('VAL/loss_epoch', patience=15, verbose=True))
    return callbacks

# %%
def _make_loggers(output_path, loggers):
    return [logger(save_dir=output_path, name='', version='') for logger in loggers]

# %%
@gin.configurable(denylist=['tfrecord', 'validation_tfrecord', 'output_path'])
def train(tfrecord, validation_tfrecord, output_path, dataset=TFIterableDataset, loggers=[TensorBoardLogger], loss=None, metrics=None, optimizer=None, batch_size=128, shuffle=None, network=None, **kwargs):
    dataloader_train = torch.utils.data.DataLoader(dataset(filepath=tfrecord, batch_size=batch_size, shuffle=shuffle), batch_size=None) #tfrecord_to_dataloader(tfrecord, batch_size=batch_size, shuffle=shuffle)
    if validation_tfrecord is not None:
        dataloader_val = torch.utils.data.DataLoader(dataset(filepath=validation_tfrecord, batch_size=batch_size, shuffle=shuffle), batch_size=None)
    else:
        dataloader_val = None

    trainer = pl.Trainer(
        default_root_dir=output_path, 
        logger=_make_loggers(output_path, loggers), 
        callbacks=_make_callbacks(output_path, validation_tfrecord is not None),
        **kwargs,
        )

    # default loss
    if loss is None:
        loss = MultinomialNLLLossFromLogits

    model = Model(network, next(iter(dataloader_train))[0], loss=loss, metrics=metrics, optimizer=optimizer)
    
    # write model summary
    with open(str(output_path / 'model.summary.txt'), 'w') as f:
        print(str(model), file=f)
    
    # write optimizer summary
    with open(str(output_path / 'optim.summary.txt'), 'w') as f:
        print(str(model.configure_optimizers()), file=f)

    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    # torch.save(model, output_path / 'model.pt') # Raises Tensorflow error during pickling (InvalidArgumentError: Cannot convert a Tensor of dtype variant to a NumPy array.)
    torch.save(model.network, output_path / 'network.pt')

    # create dummy result file
    with open(str(output_path.parent / 'result'), 'w') as f:
        pass

    # if validation_tfrecord is not None:
    #     with open(str(output_path.parent / 'result'), 'w') as f:
    #         result = trainer.validate(model, dataloader_val)[0]['VAL/loss_epoch']
    #         print(result, file=f)


# %%
@click.command()
@click.argument('tfrecord', required=True, type=str)
@click.option('--config', type=str, default=None)
@click.option('-o', '--output', required=True)
@click.option('--validation-tfrecord', type=str, default=None)
def main(tfrecord, config, output, validation_tfrecord):
    # parse gin-config config
    if config is not None:
        gin.parse_config_file(config)

    output_path = Path(f'{output}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    output_path.mkdir(parents=True)
    if config is not None:
        shutil.copy(config, str(output_path / 'config.gin'))   

    train(tfrecord, validation_tfrecord, output_path)