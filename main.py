"""
Runs a model on a single node across N-gpus.

Usage: python main.py --gpus 1
"""
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.lightning_anp import LatentModelPL
from pytorch_lightning.callbacks import EarlyStopping

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)
# The processing speed (i.e. processed batch items per second) can be lower than when the model is non-deterministic. https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LatentModelPL(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    logger = TensorBoardLogger("tensorboard_logs")
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,  # Epochs of no improvement.
        # patience=0,  # Epochs of no improvement.
        verbose=True,
        mode='min')
    if hparams.dataset == "GP":
        val_check_interval = 1.0  # Needs to be float, otherwise we're saying "check every 1 batch"
        # val_check_interval = 50  # Check every 100 batches"
    elif hparams.dataset == "smartmeter":
        val_check_interval = 0.2  # Validate every 0.2 of an epoch
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        # distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit,
        logger=logger,
        gradient_clip_val=hparams.grad_clip,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=val_check_interval,
        train_percent_check=1.0,
        val_percent_check=1.0,
        test_percent_check=1.0,
        log_save_interval=100,
        print_nan_grads=True)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)
    # ------------------------
    # 3 START TESTING
    # ------------------------
    # Just one best model here, saved by ModelCheckpoint.
    paths = pathlib.Path('.').glob('*.ckpt')
    best_model = LatentModelPL.load_from_checkpoint(next(paths))
    trainer.test(best_model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument('--gpus',
                               type=int,
                               default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed_backend',
                               type=str,
                               default='dp',
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use_16bit',
                               dest='use_16bit',
                               action='store_true',
                               help='if true uses 16 bit precision')

    # each LightningModule defines arguments relevant to it
    parser = LatentModelPL.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
