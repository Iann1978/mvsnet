import hydra
from omegaconf import DictConfig, OmegaConf
from model import get_model
from dataset import get_dataset
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from type.types import BatchedViews
from model.base_model import BaseModel
import torch.nn.functional as F
import torch
from dacite import from_dict
from dataclasses import dataclass
from typing import Any



class ModelWapper(LightningModule):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model
    
    def forward(self, x: BatchedViews):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        imgs = batch['imgs']
        targets = batch['targets']

        batched_views = BatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            images=imgs
        )
        preds = self(batched_views)
        loss = F.l1_loss(preds, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        imgs = batch['imgs']
        targets = batch['targets']

   
        batched_views = BatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            images=imgs
        )
        preds = self(batched_views)
        loss = F.l1_loss(preds, targets)
        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

@dataclass
class MainConfig:
    name: str
    dataset: Any
    model: Any


@hydra.main(config_path="../configs",
            config_name="train_unimatch_with_dtu",
            version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    cfg = from_dict(MainConfig, OmegaConf.to_container(cfg))



    dataset = get_dataset(cfg.dataset)
    print(dataset)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=False)

    model = get_model(cfg.model)
    print(model)

    model_wapper = ModelWapper(model)

    trainer = Trainer(max_epochs=10)
    trainer.fit(model_wapper, train_loader, val_loader)


if __name__ == "__main__":
    main()