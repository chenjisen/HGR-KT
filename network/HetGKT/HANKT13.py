from typing import Union

from dgl import metapath_reachable_graph
from torch import Tensor

from config.HANArgs import HANArgs
from config.KTArgs import KTArgs
from .HANKT1 import HANKT1
from .han import MetaPath, HAN
from ..DHKT.utils import get_logits_with_ec


class HANKT13(HANKT1):
    def __init__(self, args: Union[KTArgs, HANArgs]) -> None:
        super().__init__(args)

        args.han_concept_metapaths = "['c-c'], ['c-c@r']"

        self.concept_han = HAN(
            metapaths=args.han_concept_metapaths,
            in_feats=args.han_dim,
            hidden_feats=args.han_dim,
            num_heads=args.han_num_heads,
            dropout=args.han_dropout)

        self.concept_metapaths = MetaPath.get_list_from_str(args.han_concept_metapaths)
        self.concept_cgs = {mp.str: metapath_reachable_graph(self.hg, mp) for mp in self.concept_metapaths}

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
        beta_dict_0 = {}
        concept_feats = self.get_concept_feats(self.feat_dict, beta_dict_0)
        feat_dict_1 = self.feat_dict | {'c': concept_feats}
        feat_dict, beta_dict = self.embedding(feat_dict_1, exercise, student_id)  # (N, L, F)
        beta_dict |= beta_dict_0
        lstm_input = self.get_lstm_input(feat_dict, label)
        hidden = self.get_hidden(lstm_input, length)
        packed_logits = get_logits_with_ec(self.predictor, (hidden, feat_dict['e']), length)
        return packed_logits, beta_dict

    def get_concept_feats(
            self, feat_dict: dict[str, Tensor], beta_dict: dict[str, Tensor]
    ) -> Tensor:
        concept_hg_dict, concept_hg_feat, concept_beta_dict = self.concept_han(self.concept_cgs, feat_dict)
        concept_feat_flatten = concept_hg_feat['c']  # F = han_num_heads * f
        concept_feat = concept_feat_flatten.unflatten(-1, (self.han_num_heads, self.han_dim)).mean(1)
        beta_dict |= concept_beta_dict
        return concept_feat
