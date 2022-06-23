import functools
from abc import ABCMeta, abstractmethod
from typing import Any

import torch
import torch.optim.lr_scheduler
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
from torchmetrics import Accuracy, AUROC


class LitKTBase(LightningModule, metaclass=ABCMeta):
    metric: ModuleDict

    def __init__(self, backbone: Module, lr: float, threshold: float = 0.5) -> None:
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self._threshold = threshold
        self._loss_fn = torch.nn.BCEWithLogitsLoss()

        auc_f = functools.partial(AUROC, pos_label=1)
        self.metric = ModuleDict({f'{mode}_{metric}': f() for mode in ('train', 'val', 'test')
                                  for metric, f in (('acc', Accuracy), ('auc', auc_f))})

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.backbone.parameters(), self.lr)
        return optimizer

    @abstractmethod
    def shared_step(self, batch: Any, mode: str) -> Tensor:
        pass

    def shared_step_with_log(self, batch: Any, mode: str) -> Tensor:
        loss = self.shared_step(batch, mode)
        d = {k.removeprefix('train_'): v for k, v in self.metric.items() if k.startswith(mode)}
        self.log_dict(d, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self.shared_step_with_log(batch, 'train')
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss = self.shared_step_with_log(batch, 'val')
        self.log('val_loss', loss)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_step_with_log(batch, 'test')
