import hydra
from omegaconf import DictConfig, OmegaConf
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
from hydra.core.hydra_config import HydraConfig
import os
from dacite import from_dict

@dataclass
class MVSNetWapperConfig:
    # dataset: MVSNetDatasetDTUConfig
    learning_rate: float


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    device: str

@dataclass
class ModelConfig:
    learning_rate: float
    depth_steps: int
    depth_interval: float
    name: str
    in_channels: int

@dataclass
class DatasetConfig:
    training: MVSNetDatasetDTUConfig
    validation: MVSNetDatasetDTUConfig

@dataclass
class MainConfig:
    training: TrainingConfig
    model: ModelConfig
    dataset: DatasetConfig


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    print("=== Hydra Configuration ===")
    print(f"Configuration type: {type(cfg)}")
    print(f"Configuration content:")
    print(cfg)

    cfg = from_dict(MainConfig,  OmegaConf.to_container(cfg))
    print('\n')
    print(cfg)



    hydra_cfg = HydraConfig.get()
    hydra_run_dir = hydra_cfg.run.dir
    tensorboard_dir = os.path.join(hydra_run_dir, "logs")
    checkpoint_dir = os.path.join(hydra_run_dir, "checkpoints")

    



    train_dataset = MVSNetDatasetDTU(cfg.dataset.training)
    val_dataset = MVSNetDatasetDTU(cfg.dataset.validation)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=False)

    model = MVSNetWapper(cfg.model)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='mvsnet-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,  # Save top 3 best models
        save_last=True,  # Always save the last model
        every_n_epochs=1,  # Save every epoch
    )

    trainer = Trainer(max_epochs=100,
                      logger=TensorBoardLogger(tensorboard_dir),
                      callbacks=[checkpoint_callback],
                      val_check_interval=1000,
                      )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)


if __name__ == "__main__":
    main()