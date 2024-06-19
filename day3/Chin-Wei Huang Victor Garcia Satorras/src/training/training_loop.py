import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from probai.src.data.data import DataBatch
from probai.src.models.ddpm import DDPM


# Basic ML training loop
class Trainer:
    def __init__(self, model: DDPM, lr: float = 1e-3, checkpoints_path=None):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.checkpoints_path = checkpoints_path

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        self.progress_bar = tqdm(total=epochs, desc="Epochs Progress", ncols=100)

        self.val_losses = []
        self.model.to(device)
        for epoch in tqdm(range(epochs)):
            self.train_epoch(train_loader, epoch, device)
            loss = self.validate_epoch(val_loader, device)
            self.val_losses.append(loss)

            self.progress_bar.update(1)
            self.progress_bar.set_postfix(Epoch=epoch, Val_loss=loss)

            # Save checkpoint
            if self.checkpoints_path is not None:
                self.save_checkpoint(epoch)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.model.train()
        for i, batch in enumerate(train_loader):
            self.train_batch(batch, device)

    def train_batch(
        self, batch: DataBatch, device: torch.device = torch.device("cpu")
    ) -> torch.FloatTensor:
        batch = batch.to(device)
        losses = self.model.losses(
            x=batch.x,
            batch=batch.batch,
            h=batch.h,
            context=batch.context,
            edge_index=batch.edge_index,
        )
        loss = losses.mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def validate_epoch(
        self, val_loader: DataLoader, device: torch.device = torch.device("cpu")
    ) -> float:
        self.model.eval()

        all_losses = []
        for batch in val_loader:
            all_losses.append(self.validate_batch(batch, device).data)

        return torch.cat(all_losses).mean().item()

    def validate_batch(
        self, batch: DataBatch, device: torch.device = torch.device("cpu")
    ) -> torch.FloatTensor:
        batch = batch.to(device)
        losses = self.model.losses(
            x=batch.x,
            batch=batch.batch,
            h=batch.h,
            context=batch.context,
            edge_index=batch.edge_index,
        )
        return losses

    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_losses": self.val_losses,
            },
            self.checkpoints_path,
        )

    def load_checkpoint(self, checkpoint_path, device=torch.device("cpu")):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.val_losses = checkpoint["val_losses"]
        epoch = checkpoint["epoch"]
        return epoch
