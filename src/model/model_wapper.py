from pytorch_lightning import LightningModule
from .base_model import BaseModel
from ..type.types import BatchedViews
import torch.nn.functional as F
import torch
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
