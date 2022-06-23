from pathlib import Path
from typing import NamedTuple, Union

import torch
from torch import Tensor

from data.json_seq import load_seq


class Record(NamedTuple):
    """
        exercise :  (N, S),    int, 1 ~ exercise_count, 0 for padding
        label'   :  (N, S),    bool, -1 for padding
        length   :  (N)
    """
    exercise: Tensor
    label: Tensor
    length: Tensor
    combined_exercise: Tensor

    @classmethod
    def load(cls, file: Path, seq_size: int) -> 'Record':
        exercise, label, length = load_seq(file, seq_size, False)
        combined_exercise = cls.get_combined_exercise_tensor(exercise, label)
        record = cls(exercise=exercise,
                     label=label,
                     length=length,
                     combined_exercise=combined_exercise)
        return record

    @classmethod
    def get_combined_exercise_tensor(cls, exercise: Tensor, label: Tensor) -> Tensor:
        label1 = label.clone()
        label1[label1 < 0] = 0
        return cls.get_combined_exercise(exercise, label1)

    @staticmethod
    def get_combined_exercise(exercise: Union[int, Tensor], label: Union[int, Tensor]) -> Union[int, Tensor]:
        return exercise * 2 + label

    @staticmethod
    def get_exercise_from_combined(combined_exercise: Tensor) -> Tensor:
        return combined_exercise // 2

    @staticmethod
    def get_combined_feature(x: Tensor, label: Tensor) -> Tensor:
        label = label.unsqueeze(2)
        x1 = torch.cat([x * label, x * (1 - label)], dim=-1)
        return x1
