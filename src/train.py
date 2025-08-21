import hydra
from omegaconf import DictConfig, OmegaConf
from .model import get_model
from .dataset import get_dataset
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from .type.types import BatchedViews
from .model.base_model import BaseModel
import torch.nn.functional as F
import torch
from dacite import from_dict
from dataclasses import dataclass
from typing import Any
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import TensorBoardLogger
import os
import matplotlib.pyplot as plt
from .model.model_wapper import ModelWapper


@dataclass
class TrainConfig:
    name: str
    epochs: int
    batch_size: int
    float32_matmul_precision: str
    dataset: Any
    model: Any


@hydra.main(config_path="../configs",
            config_name="train_unimatch_with_dtu",
            version_base=None)
def train(cfg: DictConfig):
    print('--------------------------------train--------------------------------')
    print(OmegaConf.to_yaml(cfg))

    cfg = from_dict(TrainConfig, OmegaConf.to_container(cfg))
    hydra_cfg = HydraConfig.get()
    hydra_run_dir = hydra_cfg.run.dir
    print('run_dir: ', hydra_run_dir)
    tensorboard_dir = os.path.join(hydra_run_dir, "logs")
    checkpoint_dir = os.path.join(hydra_run_dir, "checkpoints")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


    print('get loader')
    train_dataset = get_dataset(cfg.dataset, 'val')
    val_dataset = get_dataset(cfg.dataset, 'val')
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, persistent_workers=False)
    print('train_loader: ', len(train_loader))
    print('val_loader: ', len(val_loader))


    
    print('get model')
    model = get_model(cfg.model)
    model_wapper = ModelWapper(model)

    print('get trainer')
    trainer = Trainer(max_epochs=cfg.epochs,
                      logger=TensorBoardLogger(hydra_run_dir),
                      )

    print('start training')
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)  # or use 'high'
    trainer.fit(model_wapper, train_loader, val_loader)
    print('training done')


if __name__ == "__main__":
    train()