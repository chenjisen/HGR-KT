from torch import Tensor
from torch.nn import Module

from data.data2 import Record
from .LitKT1 import BatchType, LitKT1


class LitKT2(LitKT1):
    def __init__(self, backbone: Module, lr: float) -> None:
        super().__init__(backbone, lr)

    def batch_to_args(self, batch: BatchType) -> tuple[Tensor, Tensor, Tensor]:
        r = Record(*batch)
        return r.exercise, r.combined_exercise, r.length
