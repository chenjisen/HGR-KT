import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def load_seq(file: Path, seq_size: int, leave_zero: bool) -> tuple[Tensor, Tensor, Tensor]:
    exercise_data = []
    label_data = []
    length_data = []
    with open(file) as f:
        for line in f:
            seq_line: list[list[int]] = json.loads(line)
            exercise_list = [0] * seq_size
            label_list = [-1] * seq_size
            i: int
            for i, l in enumerate(seq_line):
                exercise, label = _line_to_tuple(l, leave_zero)
                exercise_list[i] = exercise
                label_list[i] = label
            exercise_data.append(exercise_list)
            label_data.append(label_list)
            length_data.append(i + 1)
    return torch.tensor(exercise_data), torch.tensor(label_data), torch.tensor(length_data)


def load_relation(file: Path) -> dict[int, list[int]]:
    relation = {}
    with open(file) as f:
        for line in f:
            i, l = json.loads(line)
            relation[i] = l
    return relation


def _line_to_tuple(l: list[int], leave_zero: bool) -> tuple[int, int]:
    """
    # exercise_id = 0 时用作padding，因此所有exercise_id均应+1
    """
    assert len(l) == 2
    return l[0] + leave_zero, l[1]  # exercise_id, correct


def get_sample_info(dataset: Dataset) -> list[tuple[int, int]]:
    sample_info = []
    for user_index, seq_line in enumerate(dataset):
        for target_index in range(len(seq_line)):
            sample_info.append((user_index, target_index))
    return sample_info


def pad_relation(relation: dict[int, list[int]]) -> Tensor:
    """

    Args:
        relation:

    Returns:
        (N, S, M), int, 1 ~ concept_num, 0 for padding
    """
    assert 0 not in relation
    seq = [torch.tensor(v) for v in relation.values()]
    seq.insert(0, torch.tensor([]))
    padded_relation = pad_sequence(seq, True)
    return padded_relation.long()
