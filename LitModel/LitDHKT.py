from torch import Tensor
from torch.nn import Module

from .LitKT1 import LitKT1, BatchType
from .LitKT2 import LitKT2


class LitDHKT1(LitKT1):

    def __init__(self, backbone: Module, lr: float, alpha: float) -> None:
        super().__init__(backbone, lr)
        self.alpha = alpha

    def shared_step(self, batch: BatchType, mode: str) -> Tensor:
        return super().shared_step(batch, mode) + self.alpha * self.backbone.hinge_loss()


class LitDHKT2(LitKT2):

    def __init__(self, backbone: Module, lr: float, alpha: float) -> None:
        super().__init__(backbone, lr)
        self.alpha = alpha

    def shared_step(self, batch: BatchType, mode: str) -> Tensor:
        return super().shared_step(batch, mode) + self.alpha * self.backbone.hinge_loss()
