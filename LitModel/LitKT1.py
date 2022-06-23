import torch
from torch import Tensor
from torch.nn import Module

from data.data2 import Record
from network.utils import pack
from .LitKTBase import LitKTBase

BatchType = tuple[Tensor, Tensor, Tensor, Tensor]


class LitKT1(LitKTBase):
    def __init__(self, backbone: Module, lr: float) -> None:
        super().__init__(backbone, lr)

    def forward(self, *args: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.backbone(*args)
        prob = torch.sigmoid(logits).cpu().detach()
        pred = (prob >= self._threshold).long()  # shape: (batch_size, 1)
        return logits, prob, pred

    def batch_to_args(self, batch: BatchType) -> tuple[Tensor, Tensor, Tensor]:
        r = Record(*batch)
        return r.exercise, r.label, r.length

    def shared_step(self, batch: BatchType, mode: str) -> Tensor:
        r = Record(*batch)
        packed_logits, packed_prob, pred = self(*self.batch_to_args(batch))
        packed_label = pack(r.label, r.length).data
        loss = self._loss_fn(packed_logits, packed_label.float())
        packed_label = packed_label.cpu().detach()
        self.metric[f'{mode}_acc'](pred, packed_label)
        self.metric[f'{mode}_auc'](packed_prob, packed_label)
        return loss
