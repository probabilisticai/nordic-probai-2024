import numpy as np
from . import bond_analyze
from probai.src.data.mini_qm9 import DATASET_INFO

############################
# Stability and bond analysis
############################


def check_stability(positions, atom_type, dataset_info=DATASET_INFO, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info["atom_decoder"]
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype="int")

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            order = bond_analyze.get_bond_order(atom1, atom2, dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_atoms = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type_i], nr_bonds_i)
            )
        nr_stable_atoms += int(is_stable)

    molecule_stable = nr_stable_atoms == len(x)
    return molecule_stable, nr_stable_atoms, len(x)
