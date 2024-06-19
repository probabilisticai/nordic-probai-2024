import torch


def compute_edges(num_nodes):
    # Create a meshgrid of indices for the nodes in the i-th graph
    idx = torch.arange(0, num_nodes)
    row, col = torch.meshgrid(idx, idx, indexing="ij")

    # Stack the edge indices the same tensor
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0).long()
    edge_index = filter_out_diagonal_entries(edge_index)
    return edge_index


def filter_out_diagonal_entries(edge_index: torch.LongTensor) -> torch.Tensor:
    # Filter out diagonal
    row, col = edge_index
    mask = row != col
    row = row[mask]
    col = col[mask]

    # Concatenate
    return torch.stack([row, col], dim=0)


def compute_edges_squared_batch(num_nodes, num_batches):
    batch_edges = []
    for idx_batch in range(num_batches):
        edges = compute_edges(num_nodes) + num_nodes * idx_batch
        batch_edges.append(edges)

    batch_edges = torch.concatenate(batch_edges, dim=1).long()
    return batch_edges


if __name__ == "__main__":
    edge_index = compute_edges_squared_batch(5, 2)
    print(edge_index.shape)
