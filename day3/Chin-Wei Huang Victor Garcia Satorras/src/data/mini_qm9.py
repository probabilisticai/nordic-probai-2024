import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Optional
import pickle


DATASET_INFO = {
    "atom_encoder": {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4},
    "atom_decoder": ["H", "C", "N", "O", "F"],
    "n_nodes": {
        17: 13025,
        19: 13832,
        16: 10644,
        13: 3060,
        15: 7796,
        18: 13364,
        12: 1689,
        11: 807,
        14: 5136,
        7: 16,
        10: 362,
        8: 49,
        9: 124,
        4: 4,
        6: 9,
        5: 5,
        3: 1,
    },
    "max_n_nodes": 19,
    "colors_dic": ["#FFFFFF99", "C7", "C0", "C3", "C1"],
    "radius_dic": [0.46, 0.77, 0.77, 0.77, 0.77],
    "with_h": True,
    "alpha_mean": 75.3682,
    "alpha_std": 8.6825,
}


class MiniQM9Dataset(Dataset):
    """
    A dataset class for handling the MiniQM9 dataset. This class supports loading
    and storing the data as a pickle file.
    """

    def __init__(
        self,
        positions: Optional[List[torch.Tensor]] = None,
        atom_types: Optional[List[torch.Tensor]] = None,
        alphas: Optional[torch.Tensor] = None,
        file_path="raw_data/mini_qm9.pickle",
    ):
        self.file_path = file_path
        if positions is not None and atom_types is not None and alphas is not None:
            assert (
                len(positions) == len(atom_types) == len(alphas)
            ), "All arrays must have the same length"
            self.positions = positions
            self.atom_types = atom_types
            self.alphas = alphas
        else:
            self._load()
        self.edges_cache = {}

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx: int):
        position_sample = self.positions[idx]
        atom_type_sample = self.atom_types[idx]
        alpha_sample = (self.alphas[idx] - DATASET_INFO["alpha_mean"]) / DATASET_INFO[
            "alpha_std"
        ]

        # Compute edges here (replace with your edge computation logic)
        edge_index = self._compute_edges(position_sample.shape[0])

        sample = Data(
            h=torch.Tensor(atom_type_sample) * 0.25,
            x=torch.Tensor(position_sample),
            context=torch.Tensor([alpha_sample]),
            edge_index=edge_index,
        )

        return sample

    def store(self, file_path: str):
        data = {
            "positions": self.positions,
            "atom_types": self.atom_types,
            "alphas": self.alphas,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    def _load(
        self,
    ):
        with open(self.file_path, "rb") as f:
            data = pickle.load(f)
        self.positions = data["positions"]
        self.atom_types = data["atom_types"]
        self.alphas = data["alphas"]

    def _compute_edges(self, num_nodes):
        if num_nodes not in self.edges_cache:
            # Create a meshgrid of indices for the nodes in the i-th graph
            idx = torch.arange(0, num_nodes)
            row, col = torch.meshgrid(idx, idx, indexing="ij")

            # Stack the edge indices the same tensor
            edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0).long()
            edge_index = self._filter_out_diagonal_entries(edge_index)

            self.edges_cache[num_nodes] = edge_index
        return self.edges_cache[num_nodes]

    def _filter_out_diagonal_entries(
        self, edge_index: torch.LongTensor
    ) -> torch.Tensor:
        # Filter out diagonal
        row, col = edge_index
        mask = row != col
        row = row[mask]
        col = col[mask]

        # Concatenate
        return torch.stack([row, col], dim=0)
