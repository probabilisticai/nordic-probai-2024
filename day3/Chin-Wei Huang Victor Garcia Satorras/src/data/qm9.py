from torch.utils.data import DataLoader
from probai.src.data.qm9_cormorant.args import init_argparse
from probai.src.data.qm9_cormorant.collate import PreprocessQM9
from probai.src.data.qm9_cormorant.utils import initialize_datasets


def retrieve_dataloaders(
    batch_size, num_workers, raw_data="raw_data", include_charges=True
):
    # Initialize dataloader
    args = init_argparse("qm9")
    # data_dir = cfg.data_root_dir
    args, datasets, num_species, charge_scale = initialize_datasets(
        args,
        raw_data,
        "qm9",
        subtract_thermo=args.subtract_thermo,
        force_download=args.force_download,
        remove_h=False,
    )
    qm9_to_eV = {
        "U0": 27.2114,
        "U": 27.2114,
        "G": 27.2114,
        "H": 27.2114,
        "zpve": 27211.4,
        "gap": 27.2114,
        "homo": 27.2114,
        "lumo": 27.2114,
    }

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    # Construct PyTorch dataloaders from datasets
    preprocess = PreprocessQM9(load_charges=include_charges)
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=args.shuffle if (split == "train") else False,
            num_workers=num_workers,
            collate_fn=preprocess.collate_fn,
        )
        for split, dataset in datasets.items()
    }
    return dataloaders
