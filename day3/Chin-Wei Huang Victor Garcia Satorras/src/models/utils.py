from torch_geometric.utils.scatter import scatter
import torch


def center_zero(x: torch.Tensor, batch_indexes: torch.Tensor):
    assert len(x.shape) == 2 and x.shape[-1] == 3, "Dimensionality error"
    means = scatter(x, batch_indexes, dim=0, reduce="mean")
    return x - means[batch_indexes]
