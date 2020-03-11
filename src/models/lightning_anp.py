import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from src.models.model import LatentModel
from src.data.smart_meter import collate_fns, SmartMeterDataSet, get_smartmeter_df
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

    def training_step(self, batch, batch_idx):
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_pred, kl, loss, loss_mse, y_std = self.forward(
            context_x, context_y, target_x, target_y)
        tensorboard_logs = {
            "train/loss": loss,
            "train/kl": kl.mean(),
            "train/std": y_std.mean(),
            "train/mse": loss_mse.mean(),
            "train/mse": F.mse_loss(y_pred, target_y).mean(),
        }
        assert torch.isfinite(loss)
        # print('device', next(self.model.parameters()).device)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        assert all(torch.isfinite(d).all() for d in batch)
        context_x, context_y, target_x, target_y = batch
        y_pred, kl, loss, loss_mse, y_std = self.forward(
            context_x, context_y, target_x, target_y)

        tensorboard_logs = {
            "val_loss": loss,
            "val/kl": kl.mean(),
            "val/mse": loss_mse.mean(),
            "val/std": y_std.mean(),
            "val/mse": F.mse_loss(y_pred, target_y).mean(),
        }
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_end(self, outputs):
        if int(self.hparams["vis_i"]) > 0:
            # https://github.com/PytorchLightning/pytorch-lightning/blob/f8d9f8f/pytorch_lightning/core/lightning.py#L293
            loader = self.val_dataloader()[0]
            vis_i = min(int(self.hparams["vis_i"]), len(loader.dataset))
            # print('vis_i', vis_i)
            if isinstance(self.hparams["vis_i"], str):
                image = plot_from_loader(loader, self, i=int(vis_i))
                plt.show()
            else:
                image = plot_from_loader_to_tensor(loader, self, i=vis_i)
                self.logger.experiment.add_image('val/image', image,
                                                 self.trainer.global_step)

        keys = outputs[0]["log"].keys()
        # tensorboard_logs = {}
        # for k in keys:
        #     tensorboard_logs[k] = torch.stack([x["log"][k] for x in outputs if k in x["log"]]).mean()
        tensorboard_logs = {
            k: torch.stack([x["log"][k] for x in outputs
                            if k in x["log"]]).mean()
            for k in keys
        }

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        assert torch.isfinite(avg_loss)
        tensorboard_logs_str = {k: f'{v}' for k, v in tensorboard_logs.items()}
        print(f"step {self.trainer.global_step}, {tensorboard_logs_str}")

        # Log hparams with metric, doesn't work
        # self.logger.experiment.add_hparams(self.hparams.__dict__, {"avg_val_loss": avg_loss})

        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_end(self, *args, **kwargs):
        return self.validation_end(*args, **kwargs)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(),
                                  lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=2, verbose=True,
            min_lr=1e-5)  # note early stopping has patient 3
        return [optim], [scheduler]

    def _get_cache_dfs(self):
        if self._dfs is None:
            df_train, df_test = get_smartmeter_df()
            # self._dfs = dict(df_train=df_train[:600], df_test=df_test[:600])
            self._dfs = dict(df_train=df_train, df_test=df_test)
        return self._dfs

    @pl.data_loader
    def train_dataloader(self):
        df_train = self._get_cache_dfs()['df_train']
        data_train = SmartMeterDataSet(df_train, self.hparams["num_context"],
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

    @pl.data_loader
    def val_dataloader(self):
        df_test = self._get_cache_dfs()['df_test']
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

    @pl.data_loader
    def test_dataloader(self):
        df_test = self._get_cache_dfs()['df_test']
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
