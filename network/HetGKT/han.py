import ast

import torch
from dgl import DGLHeteroGraph
from dgl.nn.pytorch import GATConv
from torch import nn, Tensor
from torch.nn.functional import elu


class SemanticAttention(nn.Module):
    def __init__(self, in_feats, hidden_feats=128):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.Tanh(),
            nn.Linear(hidden_feats, 1, bias=False)
        )

    def forward(self, z):
        # z: (N, M, D), D: in_feats
        w = self.project(z)  # (N, M, 1)
        beta = torch.softmax(w, dim=1)  # (N, M, 1)
        product = beta * z  # (N, M, D)
        return product.sum(1), beta  # (N, D)


class SemanticAttention2(nn.Module):
    def __init__(self, in_feats, hidden_feats=128):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.Tanh(),
            nn.Linear(hidden_feats, 1, bias=False)
        )

    def forward(self, z):
        # z: (N, M, D), D: in_feats
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta1 = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        product = beta1 * z  # (N, M, D)
        return product.sum(1), beta  # (N, D)


class MetaPath(list):
    def __init__(self, metapath: list[str]) -> None:
        super().__init__(metapath)
        self.metapath_info = []
        for e in metapath:
            e1 = e.split('@')[0]
            src, dst = e1.split('-')
            self.metapath_info.append({'src': src, 'dst': dst, 'e': e})
        self.src = self.metapath_info[0]['src']
        self.dst = self.metapath_info[-1]['dst']
        self.tuple = tuple(metapath)
        self.str = str(metapath)

    @classmethod
    def get_list_from_str(cls, s: str) -> list['MetaPath']:
        return [cls(mp) for mp in ast.literal_eval(s)]


class HeteroHANLayer2(nn.Module):

    def __init__(self, metapaths: str, in_feats: int, out_feats: int, num_heads: int,
                 dropout: float) -> None:
        super().__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.metapaths = MetaPath.get_list_from_str(metapaths)
        self.gat_layers = nn.ModuleDict({
            mp.str: GATConv(
                in_feats, out_feats, num_heads, dropout, dropout, activation=elu,
                allow_zero_in_degree=True)
            for mp in self.metapaths
        })
        self.semantic_attentions = nn.ModuleDict({
            mp.dst: SemanticAttention2(in_feats=out_feats * num_heads)
            for mp in self.metapaths
        })

        self.num_heads = num_heads

    def forward(self, cgs: dict[str, DGLHeteroGraph], h: dict[str, nn.Embedding]
                ) -> tuple[dict[str, dict[str, Tensor]], dict[str, Tensor], dict[str, dict[str, Tensor]]]:
        semantic_embeddings = {mp.dst: {} for mp in self.metapaths}  # 'input': h[dst].repeat(1, self.num_heads)
        for mp in self.metapaths:
            layer = self.gat_layers[mp.str]
            cg = cgs[mp.str].to(h[mp.src].device)
            semantic_embedding = layer(cg, (h[mp.src], h[mp.dst])).flatten(1)  # (N, K, D) -> (N, D)
            semantic_embeddings[mp.dst][mp.str] = semantic_embedding
        semantic_embeddings_2 = {}
        betas = {}
        for k, v in semantic_embeddings.items():  # v:  [(N, D) * M]
            stacked_embedding = torch.stack(list(v.values()), dim=1)  # (N, M, D)
            semantic_attention = self.semantic_attentions[k]
            semantic_embeddings_2[k], beta = semantic_attention(stacked_embedding)  # (N, D)
            betas[k] = {k1: t.item() for k1, t in zip(v, beta)}
        return semantic_embeddings, semantic_embeddings_2, betas


class HAN(nn.Module):

    def __init__(self, metapaths, in_feats, hidden_feats, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroHANLayer2(metapaths, in_feats, hidden_feats, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HeteroHANLayer2(metapaths, hidden_feats * num_heads[l - 1],
                                               hidden_feats, num_heads[l], dropout))

    def forward(self, cgs: dict[DGLHeteroGraph], h: dict[str, Tensor]
                ) -> tuple[dict[str, dict[str, Tensor]], dict[str, Tensor], dict[str, dict[str, Tensor]]]:
        gnn: HeteroHANLayer2
        h1 = {}
        h_dict = {}
        betas = {}
        for k in h:
            h1[k] = h[k]
        for gnn in self.layers:
            h_dict, h2, betas = gnn(cgs, h1)
            for k in h2:
                h1[k] = h2[k]
        return h_dict, h1, betas
