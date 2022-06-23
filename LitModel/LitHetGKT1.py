import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from .LitKT1 import LitKT1
from .LitKT2 import LitKT2


class LitHetGKT1(LitKT1):
    def __init__(self, backbone: Module, lr: float, han_lr: float) -> None:
        super().__init__(backbone, lr)
        self.han_lr = han_lr

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.backbone.parameters(), self.lr)
        return optimizer

    def forward(self, exercise: Tensor, label: Tensor, length: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits, beta_dict = self.backbone(exercise, label, length)
        prob = torch.sigmoid(logits).cpu().detach()
        pred = (prob >= self._threshold).long()  # shape: (batch_size, 1)
        self.log_dict(beta_dict)
        return logits, prob, pred


class LitHetGKT2(LitKT2):
    def __init__(self, backbone: Module, lr: float, han_lr: float) -> None:
        super().__init__(backbone, lr)
        self.han_lr = han_lr

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.backbone.parameters(), self.lr)
        return optimizer

    def forward(self, exercise: Tensor, combined_exercise: Tensor, length: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits, beta_dict = self.backbone(exercise, combined_exercise, length)
        prob = torch.sigmoid(logits).cpu().detach()
        pred = (prob >= self._threshold).long()  # shape: (batch_size, 1)
        self.log_dict(beta_dict)
        return logits, prob, pred
