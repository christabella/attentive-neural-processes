"""PyTorch Lightning `dict` logger. Allows notebook to access self.metrics."""

from pytorch_lightning.logging import LightningLoggerBase
from pytorch_lightning.logging.tensorboard import TensorBoardLogger


class DictLogger(TensorBoardLogger):
    """PyTorch Lightning `dict` logger.
    https://pytorch-lightning.readthedocs.io/en/latest/loggers.html#custom-logger
        :param float metric: Dictionary with metric names as keys and measured quanties as values
        :param int|None step: Step number at which the metrics should be recorded
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []

    def log_hyperparams(*args, **kwargs):
        # We will do this manually with final metrics
        pass

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        self.metrics.append(metrics)
