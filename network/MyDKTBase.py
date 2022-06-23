import torch
from torch import nn, Tensor

from .utils import pack, pad


class MyDKTBase(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, exercise_count: int, seq_size: int) -> None:
        """
        Args:
            input_size: self.rnn.input_size
            hidden_size: self.rnn.hidden_size
            num_layers: self.rnn.num_layers
            exercise_count:
        """
        super().__init__()
        self.seq_size = seq_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, exercise_count + 1)

    def get_hidden(self, x: Tensor, length: Tensor) -> Tensor:
        return get_hidden(self.rnn, x, length)

    def forward(self, x: Tensor, exercise: Tensor, length: Tensor) -> Tensor:
        """
        N: batch size
        L: sequence_size

        Args:
            x: input feature after embedding layer
            exercise: (N, L)
            length: (N)
        Returns:
            model logit output, (N, 1)
        """
        hidden = self.get_hidden(x, length)
        linear_output = self.linear(hidden)
        logits = torch.gather(linear_output, -1, exercise.unsqueeze(-1)).squeeze(-1)  # TODO: check
        packed_logits = pack(logits, length).data
        return packed_logits


def get_hidden(rnn, x, length):
    packed_x = pack(x, length)
    packed_h, _ = rnn(packed_x)
    h, _ = pad(packed_h, x.shape[1])
    h1 = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), dim=1)
    return h1
