from typing import  Tuple, Union, Optional

import torch
from torch.nn import Module, Linear
from torch_geometric.typing import Adj, Tensor, OptTensor, SparseTensor
from torch_scatter import scatter_softmax, scatter

from pool import matching, utils


@torch.no_grad()
def cluster_matching(adj: SparseTensor, rank: OptTensor = None) -> Tuple[Tensor, Tensor]:
    n, device = adj.size(0), adj.device()
    row, col, val = adj.coo()

    match = matching.maximal_matching(adj, rank)
    clusters = torch.arange(n, dtype=torch.long, device=device)
    clusters[col[match]] = row[match]

    _, clusters = torch.unique(clusters, return_inverse=True)
    return clusters, match


class EdgePooling(Module):
    def __init__(self, in_channels: Optional[int] = None,
                 score: Optional[str] = 'linear',
                 score_nodes: bool = True,
                 score_activation: Optional[str] = 'sigmoid',
                 score_descending: bool = False,
                 reduce_x: str = 'sum',
                 reduce_edge: str = 'sum',
                 remove_self_loops: bool = True):
        super(EdgePooling, self).__init__()

        self.score = score
        self.score_nodes = score_nodes
        self.score_activation = score_activation
        self.score_descending = score_descending
        self.reduce_x = reduce_x
        self.reduce_edge = reduce_edge
        self.remove_self_loops = remove_self_loops

        if score == 'linear':
            self.scorer = Linear(in_channels, 1)
        elif score == 'random':
            self.scorer = lambda x: torch.rand(x.size(0), dtype=x.dtype, device=x.device)
        elif score is None:
            self.scorer = lambda x: torch.arange(x.size(0), dtype=x.dtype, device=x.device)
        else:
            self.scorer = getattr(torch, score)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None) -> Tuple[Tensor, Adj, OptTensor, OptTensor,
                                                  Union[SparseTensor, Tensor], Tensor, Tensor]:
        adj, n = edge_index, x.size(0)

        if torch.is_tensor(edge_index):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, (n, n))
        else:
            adj = edge_index

        row, col, val = adj.coo()

        if val is None:
            val = torch.ones_like(row, dtype=torch.float)

        score = self.scorer(x)

        if self.score_nodes:
            if self.score_activation not in {None, 'linear', 'softmax'}:
                score_act = getattr(torch, self.score_activation)
                score = score_act(score)

            if self.score != 'linear':
                x = score*x
        else:
            score = score[row] + score[col]

            if self.score_activation == 'softmax':
                score = scatter_softmax(score, col)
            elif self.score_activation not in {None, 'linear'}:
                score_act = getattr(torch, self.score_activation)
                score = score_act(score)

        rank = utils.get_ranking(score, descending=self.score_descending)
        cluster, match = cluster_matching(adj, rank)

        c = cluster.max() + 1

        x = scatter(x, cluster, dim=0, dim_size=c, reduce=self.reduce_x)

        if not self.score_nodes:
            matched_cluster = cluster[row[match]]
            x[matched_cluster] = x[matched_cluster]*score[match]

        adj = SparseTensor(row=cluster[row], col=cluster[col],
                           value=val, is_sorted=False,
                           sparse_sizes=(c, c)).coalesce(self.reduce_edge)

        if self.remove_self_loops:
            adj = adj.remove_diag()
        
        if torch.is_tensor(edge_index):
            row, col, edge_attr = adj.coo()
            edge_index = torch.stack([row, col])
        else:
            edge_index, edge_attr = adj, None
        
        if batch is not None:
            c_batch = torch.empty(c, dtype=torch.long, device=batch.device)
            c_batch[cluster] = batch
            batch = c_batch
        
        return x, edge_index, edge_attr, batch, cluster, match, score
