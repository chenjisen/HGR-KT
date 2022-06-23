class HANArgs:
    han_metapaths: str = "['e-s', 's-e'], ['e-c', 'c-e'], ['c-e']"
    han_lr: float = 0.005
    han_num_head = 1
    han_dim: int = 64
    han_dropout: float = 0.
    # han_weight_decay: float = 0.001
    # han_patience: int = 100
    han_c_graph_file: str = 'graph/correct_transition_graph_unweighted.json'
    han_answer_metapaths: str = "['e-ce'], ['a-ce']"
