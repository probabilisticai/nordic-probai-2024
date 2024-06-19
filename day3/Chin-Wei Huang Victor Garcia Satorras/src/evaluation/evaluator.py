from probai.src.models.ddpm import DDPM
from probai.src.models.utils import center_zero
from probai.src.evaluation.stability_analyze import check_stability
from probai.src.evaluation.visualizer import plot_data3d
from typing import Tuple
from torch_geometric.loader import DataLoader
import torch
import numpy as np


class Evaluator:
    def __init__(self, ddpm: DDPM, valid_loader: DataLoader):
        self.ddpm = ddpm
        self.dataloader = valid_loader

    def sample_batch(self, device=torch.device("cpu")):
        print(f"Generating a batch of {self.dataloader.batch_size} samples.")
        self.ddpm.to(device)
        for batch_data in self.dataloader:
            batch_data = batch_data.to(device)
            shape = [
                batch_data.x.shape[0],
                batch_data.x.shape[1] + batch_data.h.shape[1],
            ]
            xh = self.ddpm.sample(
                shape, edge_index=batch_data.edge_index, batch=batch_data.batch
            )
            xh = torch.Tensor(xh)
            break

        x = xh[:, 0:3]
        x = center_zero(x, batch_data.batch.cpu())
        h = xh[:, 3:]
        return x, h, batch_data.ptr.cpu()

    def eval_stability(self, x, h, ptr) -> Tuple[float, float]:
        print(f"Evaluating stability on {len(ptr) - 1} samples")
        st_dict = {
            "num_stable_mols": 0,
            "num_mols": 0,
            "num_stable_atoms": 0,
            "num_atoms": 0,
        }

        for idx in range(len(ptr) - 1):
            p1, p2 = ptr[idx], ptr[idx + 1]
            x_sample = x[p1:p2]
            h_sample = h[p1:p2]
            mol_stable, num_stable_atoms, num_atoms = check_stability(
                x_sample, h_sample
            )

        st_dict["num_stable_mols"] += mol_stable
        st_dict["num_mols"] += 1
        st_dict["num_stable_atoms"] += num_stable_atoms
        st_dict["num_atoms"] += num_atoms

        atom_st = st_dict["num_stable_atoms"] / st_dict["num_atoms"]
        mol_st = st_dict["num_stable_mols"] / st_dict["num_mols"]

        return atom_st, mol_st

    def eval_plot(
        self, x: np.ndarray, h: np.ndarray, ptr: np.ndarray, max_num_plots: int = 5
    ):
        eval_plot(x, h, ptr, max_num_plots)


def eval_plot(
    x: np.ndarray,
    h: np.ndarray,
    ptr: np.ndarray,
    max_num_plots: int = 5,
    sphered_3d=True,
):
    # Plot a
    for idx in range(len(ptr) - 1):
        p1, p2 = ptr[idx], ptr[idx + 1]
        x_sample = x[p1:p2]
        h_sample = h[p1:p2]
        plot_data3d(
            x_sample, h_sample, spheres_3d=sphered_3d, camera_elev=0, camera_azim=0
        )

        if idx >= max_num_plots - 1:
            break
