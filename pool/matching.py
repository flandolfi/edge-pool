import torch
from torch_geometric.typing import OptTensor, Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_min

from pool import utils


@torch.no_grad()
@torch.jit.script
def maximal_matching(adj: SparseTensor, rank: OptTensor = None) -> Tensor:
    n, m, device = adj.size(0), adj.nnz(), adj.device()
    row, col, val = adj.coo()

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    if rank.size(0) == n:
        rank = utils.rank_edge_from_nodes(adj, rank)
    
    match = torch.zeros(m, dtype=torch.bool, device=device)
    max_rank = torch.full((n,), fill_value=n*n, dtype=torch.long, device=device)
    mask = row < col
    
    while mask.any():
        row_rank = max_rank.clone()
        col_rank = max_rank.clone()
        scatter_min(rank[mask], row[mask], out=row_rank)
        scatter_min(rank[mask], col[mask], out=col_rank)
        node_rank = torch.minimum(row_rank, col_rank)
        edge_rank = torch.minimum(node_rank[row], node_rank[col])

        match = match | torch.eq(rank, edge_rank)

        unmatched = torch.ones(n, dtype=torch.bool, device=device)
        idx = torch.cat([row[match], col[match]], dim=0)
        unmatched[idx] = False
        mask = mask & unmatched[row] & unmatched[col]
    
    return match

