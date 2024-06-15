from pytorch_lightning import Trainer
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from depth_cov.data.data_modules import DepthCovDataModule
import time

def validate_model(batch_size, model_path):
    data_module = DepthCovDataModule(batch_size)

    model = NonstationaryGpModule.load_from_checkpoint(model_path)

    trainer = Trainer(accelerator='gpu', devices=1)

    trainer.validate(model, datamodule=data_module)

if __name__ == "__main__":
    batch_size = 4
    # model_path = "./models/gp-epoch=97-loss_val=-60.5360.ckpt"///
    model_path = "./models/gp-epoch=283-loss_val=-62.0410.ckpt"
    validate_model(batch_size, model_path)
