from typing import Union

from dgl import metapath_reachable_graph
from torch import Tensor, nn

from config.HANArgs import HANArgs
from config.KTArgs import KTArgs
from data.Record import Record
from .HANKT1 import HANKT1, get_sub_cgs
from .han import MetaPath, HAN
from ..DHKT.utils import get_logits_with_ec


class HANKT12(HANKT1):
    def __init__(self, args: Union[KTArgs, HANArgs]) -> None:
        han_out_dim = args.han_dim * args.han_num_heads[-1]
        args.question_dim = han_out_dim
        super().__init__(args)

        self.answer_embedding = nn.Embedding(2, args.han_dim)
        self.feat_dict['a'] = self.answer_embedding.weight

        self.answer_han = HAN(
            metapaths=args.han_answer_metapaths,
            in_feats=args.han_dim,
            hidden_feats=args.han_dim,
            num_heads=args.han_num_heads,
            dropout=args.han_dropout)

        self.answer_metapaths = MetaPath.get_list_from_str(args.han_answer_metapaths)
        self.answer_cgs = {mp.str: metapath_reachable_graph(self.hg, mp) for mp in self.answer_metapaths}

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
        combined_exercise = Record.get_combined_exercise(exercise, label)
        lstm_input = self.get_lstm_input2(feat_dict, beta_dict, combined_exercise)
        hidden = self.get_hidden(lstm_input, length)
        packed_logits = get_logits_with_ec(self.predictor, (hidden, feat_dict['e']), length)
        return packed_logits, beta_dict

    def get_lstm_input2(
            self, feat_dict: dict[str, Tensor], beta_dict: dict[str, Tensor], combined_exercise: Tensor
    ) -> Tensor:
        combined_exercise_flatten = combined_exercise[combined_exercise > 0].cpu()
        answer_sub_cgs = get_sub_cgs(self.answer_cgs, self.answer_metapaths, self.node_dict,
                                     combined_exercise_flatten, ['ce'])
        exercise_feat_flatten = feat_dict['e_all']
        exercise_feat: Tensor = exercise_feat_flatten.unflatten(-1, (self.han_num_heads, self.han_dim))
        exercise_feat = exercise_feat.mean(1)
        answer_feat_dict = self.feat_dict | {'e': exercise_feat}
        answer_hg_dict, answer_hg_feat, answer_beta_dict = self.answer_han(
            answer_sub_cgs, answer_feat_dict)  # F = han_num_heads * f
        combined_exercise_feat_x = answer_hg_feat['ce'][combined_exercise]  # (N, L, F)
        beta_dict |= answer_beta_dict
        return combined_exercise_feat_x
