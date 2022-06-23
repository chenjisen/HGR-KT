from typing import Union, Container

import dgl
import torch
from dgl import DGLHeteroGraph, metapath_reachable_graph
from torch import nn, Tensor

from config.HANArgs import HANArgs
from config.KTArgs import KTArgs
from .han import MetaPath, HAN
from ..DHKT.utils import get_logits_with_ec
from ..MyDKTBase import MyDKTBase


class HANKT1(MyDKTBase):
    hg: DGLHeteroGraph

    def __init__(self, args: Union[KTArgs, HANArgs]) -> None:
        han_out_dim = args.han_dim * args.han_num_heads[-1]
        if not hasattr(args, 'question_dim'):
            args.question_dim = han_out_dim * 2
        super().__init__(args.question_dim, args.hidden_dim, args.num_layers,
                         args.exercise_count,
                         args.seq_size)
        self.predictor = nn.Sequential(
            nn.Linear(args.hidden_dim + han_out_dim, args.hidden_dim * 3),
            nn.Tanh(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.hidden_dim * 3, 1)
        )

        self.student_embedding = nn.Embedding(args.student_count, args.han_dim)
        self.exercise_embedding = nn.Embedding(args.exercise_count + 1, args.han_dim, padding_idx=0)
        self.combined_exercise_embedding = nn.Embedding(args.exercise_count * 2 + 2, args.han_dim, padding_idx=0)
        self.concept_embedding = nn.Embedding(args.concept_count + 1, args.han_dim, padding_idx=0)
        self.answer_embedding = nn.Embedding(2, han_out_dim)

        self.feat_dict = {
            # 's': self.student_embedding.weight,
            'e': self.exercise_embedding.weight,
            'ce': self.combined_exercise_embedding.weight,
            'c': self.concept_embedding.weight,
            'a': self.answer_embedding.weight,
            'ae': self.combined_exercise_embedding.weight
        }

        self.han = HAN(
            metapaths=args.han_metapaths,
            in_feats=args.han_dim,
            hidden_feats=args.han_dim,
            num_heads=args.han_num_heads,
            dropout=args.han_dropout)
        self.han_dim = args.han_dim
        self.han_num_heads = args.han_num_heads[0]

        self.hg = args.hg
        self.metapaths = MetaPath.get_list_from_str(args.han_metapaths)
        self.cgs = {mp.str: metapath_reachable_graph(self.hg, mp) for mp in self.metapaths}
        self.node_dict = {k: torch.arange(v.shape[0]) for k, v in self.feat_dict.items()}

    def forward(self, exercise: Tensor, label: Tensor, length: Tensor, student_id: Tensor = None
                ) -> tuple[Tensor, dict]:
        """
        N: batch size
        L: sequence_size
        f: han_dim
        F: han_out_feats

        Args:
            exercise: (N, L)
            label: (N, L)
            length: (N)
            student_id: (N)
        Returns:
            model logit output, (N, 1)
        """
        feat_dict, beta_dict = self.embedding(self.feat_dict, exercise, student_id)  # (N, L, F)
        lstm_input = self.get_lstm_input(feat_dict, label)
        hidden = self.get_hidden(lstm_input, length)
        packed_logits = get_logits_with_ec(self.predictor, (hidden, feat_dict['e']), length)
        return packed_logits, beta_dict

    def get_lstm_input(self, feat_dict: dict[str, Tensor], label: Tensor) -> Tensor:
        answer_feat = self.answer_embedding(label.clamp(0))
        lstm_input = torch.cat((feat_dict['e'], answer_feat), dim=-1)
        # lstm_input = Record.get_combined_feature(exercise_feat, label)  # (N, L, 2F)
        return lstm_input

    def embedding(self, feat_dict, exercise: Tensor, student_id: Tensor) -> tuple[dict, dict]:
        sub_cgs = self.get_sub_cgs(exercise, ['e'], ['e'])
        hg_dict, hg_feat, beta_dict = self.han(sub_cgs, feat_dict)  # F = han_num_heads * f
        # student_feats_2 = hg_feat['s']

        exercise_feat = hg_feat['e']
        exercise_feat_x = exercise_feat[exercise]  # (N, L, F)

        # n, l = combined_exercise.shape
        # if student_id is not None:
        #     student_feat_x_0 = student_feats_2[student_id]
        # else:
        #     student_feat_x_0 = student_feats_2.mean(dim=0).expand(n, 1, -1).clone()
        # student_feat_x_0 -= student_feats_2.mean(dim=0)  # (N, 1, F)
        # student_feat_x = student_feat_x_0.expand(-1, l, -1)  # (N, L, F)

        feat_dict = {
            # 's': student_feat_x,
            'e': exercise_feat_x,
            'e_all': exercise_feat
        }
        return feat_dict, beta_dict

    def get_sub_cgs(
            self, exercise: Tensor, reduced_types: Container[str], add_self_loop_types: Container[str]
    ) -> dict[str, DGLHeteroGraph]:
        exercise_flatten = exercise[exercise > 0].cpu()
        return get_sub_cgs(self.cgs, self.metapaths, self.node_dict,
                           exercise_flatten, reduced_types, add_self_loop_types)


def get_sub_cgs(
        cgs: dict[str, DGLHeteroGraph], metapaths: list[MetaPath], node_dict: dict[str, Tensor],
        dst_nodes: Tensor, reduced_types: Container[str] = (), add_self_loop_types: Container[str] = ()
) -> dict[str, DGLHeteroGraph]:
    sub_cgs = {}

    for mp in metapaths:
        cg = cgs[mp.str]
        if mp.dst in reduced_types:
            data_dict = {(mp.src, '_E2', mp.dst): cg.in_edges(dst_nodes),
                         (mp.src, 'self-loop', mp.src): (node_dict[mp.src],) * 2,
                         (mp.dst, 'self-loop', mp.dst): (node_dict[mp.dst],) * 2}
            cg = dgl.heterograph(data_dict).edge_type_subgraph(['_E2'])
        if mp.src == mp.dst and mp.dst in add_self_loop_types:
            dgl.add_self_loop(cg)
        sub_cgs[mp.str] = cg
    return sub_cgs
