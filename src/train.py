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
        masks = batch['masks']

        batched_views = BatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            images=imgs
        )
        preds = self(batched_views)
        preds = preds * masks
        targets = targets * masks
        loss = F.l1_loss(preds, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        imgs = batch['imgs']
        targets = batch['targets']
        masks = batch['masks']

   
        batched_views = BatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            images=imgs
        )
        preds = self(batched_views)
        preds = preds * masks
        targets = targets * masks
        loss = F.l1_loss(preds, targets)
        self.log('val_loss', loss)

        if batch_idx == 0:
            groundtruth_normalized = ((targets[0,0].cpu() - 500) / (1000 - 500)).clamp(0, 1).cpu()
            groundtruth_normalized_colored = self.apply_colormap(groundtruth_normalized)


            preds_colored = self.apply_colormap(preds[0,0].cpu())
            preds_normalized = ((preds[0,0].cpu() - 500) / (1000 - 500)).clamp(0, 1).cpu()
            preds_normalized_colored = self.apply_colormap(preds_normalized)
            
            self.logger.experiment.add_image('depth/ref_img', imgs[0][0].cpu(), self.global_step, dataformats='CHW')
            self.logger.experiment.add_image('depth/src_img', imgs[0][1].cpu(), self.global_step, dataformats='CHW')
            self.logger.experiment.add_image('depth/groundtruth_normalized', groundtruth_normalized, self.global_step)
            self.logger.experiment.add_image('depth/groundtruth_normalized_colored', groundtruth_normalized_colored, self.global_step)
            self.logger.experiment.add_image('depth/preds_colored', preds_colored, self.global_step)
            self.logger.experiment.add_image('depth/preds_normalized', preds_normalized, self.global_step)
            self.logger.experiment.add_image('depth/preds_normalized_colored', preds_normalized_colored, self.global_step)
            self.logger.experiment.add_image('depth/mask', masks[0,0].cpu(), self.global_step, dataformats='CHW')

            if hasattr(self.model, "warped_feature1"):
                warped_feature1 = self.model.warped_feature1[0,:3,0]
                self.logger.experiment.add_image('depth/warped_feature1', warped_feature1, self.global_step)
                warped_feature1 = self.model.warped_feature1[0,[0],0]
                min, max = torch.min(warped_feature1), torch.max(warped_feature1)
                warped_feature1 = (warped_feature1 - min) / (max - min)
                warped_feature1_colored = self.apply_colormap(warped_feature1.cpu())
                self.logger.experiment.add_image('depth/warped_feature1_colored', warped_feature1_colored, self.global_step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
    def apply_colormap(self, depth_tensor):
        """Apply colormap to depth tensor for better visualization"""
        # Convert to numpy and normalize
        depth_np = depth_tensor.squeeze().numpy()
        depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        
        # Apply colormap (e.g., 'viridis', 'plasma', 'inferno', 'magma')
        colored = plt.cm.viridis(depth_normalized)
        
        # Convert back to tensor (remove alpha channel)
        colored_tensor = torch.from_numpy(colored[:, :, :3]).permute(2, 0, 1)
        return colored_tensor

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