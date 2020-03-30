"""PyTorch Lightning Module that wraps a Neural Process specified by LatentModel."""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from src.models.model import LatentModel
from src.data.smart_meter import collate_fns, SmartMeterDataSet, get_smartmeter_df
from src.data.gp_sine_curves_dataset import generate_GP_data, GPCurvesDataset, collate_fns_GP
from src.plot import plot_from_loader_to_tensor, plot_from_loader
from src.utils import ObjectDict
from matplotlib import pyplot as plt


class LatentModelPL(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = ObjectDict()
        self.hparams.update(
            hparams.__dict__ if hasattr(hparams, '__dict__') else hparams)
        self.model = LatentModel(**self.hparams)
        self._dfs = None

    def forward(self, context_x, context_y, target_x, target_y):
        return self.model(context_x, context_y, target_x, target_y)

    def batch_step(self, batch, stage='train'):
        """Batch is taken from the relevant dataloader, depending on stage.
        """
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_pred, kl, loss, loss_mse, y_std = self.forward(
            context_x, context_y, target_x, target_y)

        tensorboard_logs = {
            f"{stage}_loss": loss,
            f"{stage}_kl": kl.mean(),
            f"{stage}_std": y_std.mean(),
            f"{stage}_mse": loss_mse.mean(),
            f"{stage}_mse_functional": F.mse_loss(y_pred, target_y).mean(),
            f"{stage}_y_pred": y_pred.mean(),
            f"{stage}_context_x": context_x.mean(),
            f"{stage}_context_y": context_y.mean(),
            f"{stage}_target_x": target_x.mean(),
            f"{stage}_target_y": target_y.mean()
        }
        assert torch.isfinite(loss)
        return loss, tensorboard_logs

    def training_step(self, batch, batch_idx):
        loss, tensorboard_logs = self.batch_step(batch, stage='train')
        PLOT_INTERVAL = 100
        if batch_idx % PLOT_INTERVAL == 0:
            loader = self.train_dataloader()
            image = plot_from_loader_to_tensor(loader,
                                               self,
                                               i=self.hparams["vis_i"])
            # https://github.com/PytorchLightning/pytorch-lightning/blob/f8d9f8f/pytorch_lightning/core/lightning.py#L293
            self.logger.experiment.add_image('train_image', image,
                                             self.trainer.global_step)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, tensorboard_logs = self.batch_step(batch, stage='val')
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """Outputs are a list defined by validation_step().
        Early stopping will be enabled if ‘val_loss’ is found in return dict.
        https://pytorch-lightning.readthedocs.io/en/latest/early_stopping.html
        """
        # loader is torch.utils.data.DataLoader
        loader = self.val_dataloader()
        image = plot_from_loader_to_tensor(loader,
                                           self,
                                           i=self.hparams["vis_i"])
        # https://github.com/PytorchLightning/pytorch-lightning/blob/f8d9f8f/pytorch_lightning/core/lightning.py#L293
        self.logger.experiment.add_image('val_image', image,
                                         self.trainer.global_step)

        keys = outputs[0]["log"].keys()
        tensorboard_logs = {
            k: torch.stack([x["log"][k] for x in outputs]).mean()
            for k in keys
        }
        # Average over all batches (outputs is a list of all batch outputs).
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        assert torch.isfinite(avg_loss)
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, tensorboard_logs = self.batch_step(batch, stage='test')
        return {"test_loss": loss, "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        loader = self.test_dataloader()
        image = plot_from_loader(loader, self, i=self.hparams["vis_i"])
        plt.savefig('test_plot.pgf')

        image = plot_from_loader_to_tensor(loader,
                                           self,
                                           i=self.hparams["vis_i"])
        self.logger.experiment.add_image('test_image', image,
                                         self.trainer.global_step)
        keys = outputs[0]["log"].keys()
        tensorboard_logs = {
            k: torch.stack([x["log"][k] for x in outputs]).mean()
            for k in keys
        }
        # Average over all batches (outputs is a list of all batch outputs).
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        assert torch.isfinite(avg_loss)

        return {
            "test_loss": avg_loss,
            "progress_bar":
            tensorboard_logs,  # Print test_* scalars in stdout.
            # Unfortunately this means these test_* scalars will be logged in
            # TB even though they're just a single point, but we need this so
            # guild compare can access them (or at least just loss and MSE).
            "log": {
                'test_loss': tensorboard_logs['test_loss'],
                'test_mse_functional': tensorboard_logs['test_mse_functional']
            },
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(),
                                  lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=2, verbose=True,
            min_lr=1e-5)  # note early stopping has patient 3
        return [optim], [scheduler]

    def prepare_data(self):
        """https://pytorch-lightning.readthedocs.io/en/0.7.1/lightning-module.html"""
        if self.hparams["dataset"] == "smartmeter":
            self._df_train, self._df_test = get_smartmeter_df()
        elif self.hparams["dataset"] == "GP":
            self._train_X, self._train_F = generate_GP_data(num_functions=1000,
                                                            num_samples=100)
            self._test_X, self._test_F = generate_GP_data(num_functions=200,
                                                          num_samples=100)

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams["dataset"] == "smartmeter":
            df_train = self._df_train
            data_train = SmartMeterDataSet(df_train,
                                           self.hparams["num_context"],
                                           self.hparams["num_extra_target"])
            return torch.utils.data.DataLoader(
                data_train,
                batch_size=self.hparams["batch_size"],
                shuffle=True,
                collate_fn=collate_fns(
                    self.hparams["num_context"],
                    self.hparams["num_extra_target"],
                    sample=True,
                    context_in_target=self.hparams["context_in_target"]),
                num_workers=self.hparams["num_workers"],
            )
        elif self.hparams["dataset"] == "GP":
            data_train = GPCurvesDataset(self._train_X, self._train_F)
            return torch.utils.data.DataLoader(
                data_train,
                batch_size=self.hparams["batch_size"],
                shuffle=True,
                collate_fn=collate_fns_GP(
                    self.hparams["num_context"],
                    context_in_target=self.hparams["context_in_target"]),
                num_workers=self.hparams["num_workers"],
            )

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams["dataset"] == "smartmeter":
            df_test = self._df_test
            data_test = SmartMeterDataSet(df_test, self.hparams["num_context"],
                                          self.hparams["num_extra_target"])
            return torch.utils.data.DataLoader(
                data_test,
                batch_size=self.hparams["batch_size"],
                shuffle=False,
                collate_fn=collate_fns(
                    self.hparams["num_context"],
                    self.hparams["num_extra_target"],
                    sample=False,
                    context_in_target=self.hparams["context_in_target"]),
            )
        elif self.hparams["dataset"] == "GP":
            data_test = GPCurvesDataset(self._test_X, self._test_F)
            return torch.utils.data.DataLoader(
                data_test,
                batch_size=self.hparams["batch_size"],
                shuffle=True,
                collate_fn=collate_fns_GP(
                    self.hparams["num_context"],
                    context_in_target=self.hparams["context_in_target"]),
                num_workers=self.hparams["num_workers"],
            )

    @pl.data_loader
    def test_dataloader(self):
        if self.hparams["dataset"] == "smartmeter":
            df_test = self._df_test
            data_test = SmartMeterDataSet(df_test, self.hparams["num_context"],
                                          self.hparams["num_extra_target"])
            return torch.utils.data.DataLoader(
                data_test,
                batch_size=self.hparams["batch_size"],
                shuffle=False,
                collate_fn=collate_fns(
                    self.hparams["num_context"],
                    self.hparams["num_extra_target"],
                    sample=False,
                    context_in_target=self.hparams["context_in_target"]),
            )
        elif self.hparams["dataset"] == "GP":
            data_test = GPCurvesDataset(self._test_X, self._test_F)
            return torch.utils.data.DataLoader(
                data_test,
                batch_size=self.hparams["batch_size"],
                shuffle=True,
                collate_fn=collate_fns_GP(
                    self.hparams["num_context"],
                    context_in_target=self.hparams["context_in_target"]),
                num_workers=self.hparams["num_workers"],
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--latent_enc_self_attn_type',
                            type=str,
                            default='uniform',
                            help='Attention type')
        parser.add_argument('--det_enc_self_attn_type',
                            type=str,
                            default='uniform',
                            help='Attention type')
        parser.add_argument('--det_enc_cross_attn_type',
                            type=str,
                            default='multihead',
                            choices=['uniform', 'ptmultihead', 'multihead'],
                            help='Makes an NP an ANP.')
        parser.add_argument('--dataset',
                            type=str,
                            default='GP',
                            choices=['GP', 'smartmeter'])
        parser.add_argument('--learning_rate',
                            type=float,
                            default=1e-2,
                            help='')
        parser.add_argument('--hidden_dim', type=int, default=128, help='')
        parser.add_argument('--latent_dim', type=int, default=128, help='')
        parser.add_argument('--attention_layers', type=int, default=2, help='')
        parser.add_argument('--n_latent_encoder_layers',
                            type=int,
                            default=2,
                            help='')
        parser.add_argument('--n_det_encoder_layers',
                            type=int,
                            default=2,
                            help='')
        parser.add_argument('--n_decoder_layers', type=int, default=2, help='')
        parser.add_argument('--dropout', type=int, default=0, help='')
        parser.add_argument('--attention_dropout',
                            type=int,
                            default=0,
                            help='')
        parser.add_argument('--batchnorm', action='store_true', help='')
        # True by default
        parser.add_argument('--use_deterministic_path',
                            action='store_false',
                            help='')
        parser.add_argument('--use_self_attn', action='store_false', help='')
        parser.add_argument('--context_in_target',
                            action='store_true',
                            help='')
        # False by default
        parser.add_argument('--use_lvar', action='store_true', help='')
        parser.add_argument('--use_rnn', action='store_true', help='')

        parser.add_argument('--min_std', type=float, default=0.005, help='')
        parser.add_argument('--grad_clip',
                            type=int,
                            default=0,
                            help='0 means no clipping.')
        # TODO: This should vary among functions/meta-datasets?
        parser.add_argument('--num_context', type=int, default=24 * 4, help='')
        parser.add_argument('--num_extra_target',
                            type=int,
                            default=24 * 4,
                            help='')
        parser.add_argument('--max_nb_epochs', type=int, default=10, help='')
        parser.add_argument('--num_workers', type=int, default=3, help='')
        parser.add_argument('--batch_size', type=int, default=16, help='')
        parser.add_argument('--num_heads', type=int, default=8, help='')
        parser.add_argument('--x_dim',
                            type=int,
                            default=1,
                            help='Should be 17 if dataset is smartmeter.')
        parser.add_argument('--y_dim', type=int, default=1, help='')
        parser.add_argument('--vis_i', type=int, default=670, help='')

        return parser
