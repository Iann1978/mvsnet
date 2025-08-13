import hydra
from omegaconf import DictConfig
from MVSNetDatasetDTU import MVSNetDatasetDTU, MVSNetDatasetDTUConfig
from MVSNetWapper import MVSNetWapper
from pytorch_lightning import LightningModule
from dataclasses import dataclass
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import torch.nn.functional as F
import torch

@dataclass
class MVSNetWapperConfig:
    # dataset: MVSNetDatasetDTUConfig
    learning_rate: float


def get_data_loaders():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)
    return train_loader, val_loader

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    print("=== Hydra Configuration ===")
    print(f"Configuration type: {type(cfg)}")
    print(f"Configuration content:")
    print(cfg)

    train_dataset = MVSNetDatasetDTU(cfg.dataset.validation)
    val_dataset = MVSNetDatasetDTU(cfg.dataset.validation)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)

    model = MVSNetWapper(cfg.model)
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='mvsnet-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,  # Save top 3 best models
        save_last=True,  # Always save the last model
        every_n_epochs=1  # Save every epoch
    )

    trainer = Trainer(max_epochs=10, logger=TensorBoardLogger('lightning_logs'), callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)


if __name__ == "__main__":
    main()