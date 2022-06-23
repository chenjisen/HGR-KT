from typing import Union, Dict, Tuple, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

BatchType = Dict[str, Union[List, Tensor]]


class LitKT(pl.LightningModule):

    def __init__(self, backbone: Module, lr: float, warmup: int) -> None:
        super().__init__()
        self.backbone = backbone
        self._loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.lr = lr
        self.warmup = warmup
        self.warmup_c = backbone.d_model ** (-0.5)
        self._threshold = 0.5

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict]]:
        optimizer = torch.optim.Adam(self.backbone.parameters(), self.lr)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, self.warm_decay),
            "interval": "step",
            "frequency": 1,
            "name": "Noam Optimizer"
        }

        return [optimizer], [scheduler]

    def warm_decay(self, step: int) -> float:
        if step < self.warmup:
            return self.warmup_c * step / self.warmup ** 1.5
        else:
            return self.warmup_c * step ** -0.5

    def forward(self, batch: BatchType) -> Tuple[Tensor, Tensor]:
        output = self.backbone(batch['input'], batch['target_id'])
        pred = (torch.sigmoid(output) >= self._threshold).long()  # shape: (batch_size, 1)
        return output, pred

    def shared_step(self, batch: BatchType, mode: str) -> Dict[str, Union[Tensor, float]]:
        label = batch['label']  # shape: (batch_size, 1)
        model_out, pred = self.forward(batch)
        batch_loss = self._loss_fn(model_out, label.float()).mean()

        num_corrects = torch.eq(pred, label).sum().item()
        num_total = len(label)

        acc = num_corrects / num_total
        try:
            auc = roc_auc_score(label.cpu().detach(), model_out.cpu().detach())
        except ValueError:
            auc = -1
        out = {f'{mode}_loss': batch_loss, f'{mode}_acc': acc, f'{mode}_auc': auc}
        return out

    def training_step(self, batch: BatchType, batch_idx: int) -> STEP_OUTPUT:
        out = self.shared_step(batch, 'train')
        self.log_dict({'acc': out['train_acc'], 'auc': out['train_auc']}, prog_bar=True)
        return out['train_loss']

    def validation_step(self, batch: BatchType, batch_idx: int) -> None:
        out = self.shared_step(batch, 'val')
        self.log_dict(out, prog_bar=True)

    def test_step(self, batch: BatchType, batch_idx: int) -> None:
        out = self.shared_step(batch, 'test')
        self.log_dict(out)
