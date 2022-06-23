import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


def get_prob_weight_mean(prob_w: Tensor) -> Tensor:
    prob_w_mean = (prob_w[2::2] + prob_w[3::2]) / 2
    return torch.cat((torch.zeros_like(prob_w_mean[:1]), prob_w_mean))


def pad(seq: PackedSequence, seq_size: int) -> tuple[Tensor, Tensor]:
    return pad_packed_sequence(seq, batch_first=True, total_length=seq_size)


def pack(seq: Tensor, lens: Tensor) -> PackedSequence:
    return pack_padded_sequence(seq, lens.cpu(), batch_first=True, enforce_sorted=False)
