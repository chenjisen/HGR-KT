from typing import Optional

import torch
from torch import Tensor, nn


def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None) -> Tensor:
    weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
    nn.init.normal_(weight)
    if padding_idx is not None:
        with torch.no_grad():
            weight[padding_idx].fill_(0)
    return weight
