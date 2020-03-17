"""
Runs a model on a single node across N-gpus.

Usage: python main.py --gpus 1
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.logging.tensorboard import TensorBoardLogger
from src.models.lightning_anp import LatentModelPL

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


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
    trainer = pl.Trainer(max_epochs=hparams.epochs,
                         gpus=hparams.gpus,
                         distributed_backend=hparams.distributed_backend,
                         use_amp=hparams.use_16bit)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    logger = TensorBoardLogger("tensorboard_logs")
    trainer.fit(model, logger=logger)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument('--gpus',
                               type=int,
                               default=2,
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
