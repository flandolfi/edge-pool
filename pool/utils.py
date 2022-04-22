import torch
from torch_geometric.typing import Tensor, Union, SparseTensor


def rank_edge_from_nodes(edge_index: Union[SparseTensor, Tensor], rank: Tensor) -> Tensor:
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index[0], edge_index[1]

    n = rank.size(0)
    row_rank, col_rank = rank[row], rank[col]
    return torch.minimum(row_rank*n + col_rank, col_rank*n + row_rank)


def get_ranking(value: Tensor, descending: bool = True) -> Tensor:
    perm = torch.argsort(value.view(-1), 0, descending)
    rank = torch.zeros_like(perm)
    rank[perm] = torch.arange(rank.size(0), dtype=torch.long, device=rank.device)

    return rank
