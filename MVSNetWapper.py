from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_lightning import LightningModule
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra
from typing_extensions import TypedDict
from jaxtyping import Float, Int64
from torch import Tensor
from einops import rearrange
import matplotlib.pyplot as plt


class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    images: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width


@dataclass
class MVSNetWapperConfig:
    # dataset: MVSNetDatasetDTUConfig
    learning_rate: float
    depth_steps: int
    depth_interval: float

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        # x (B, V, 3, H, W)
        B, V, C, H, W = x.shape
        x = x.view(-1,C, H, W) # (B*V, 3, H, W)
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        x = x.view(B, V, 32, H//4, W//4) # (B, V, 32, H/4, W/4)
        return x

class HomographyWarping():
    def __init__(self):
        pass

    def __call__(self, intrinsics, extrinsics, feats, deps):
        return self.forward(intrinsics, extrinsics, feats, deps)
    
    def forward(self, intrinsics, extrinsics, feats, deps):
        # intrinsics (B, V, 3, 3)
        # extrinsics (B, V, 4, 4)
        # feats (B, V, 32, H/4, W/4)
        # return (B, D, 32, H/4, W/4)
        B, V, C, H, W = feats.shape
        D = deps.shape[0]
       
        volume_sum = torch.zeros(D, C, H, W, device=feats.device, dtype=feats.dtype)
        volume_sq_sum = torch.zeros(D, C, H, W, device=feats.device, dtype=feats.dtype)

        for v in range(V):
            volume = self.homography_warp1(v, intrinsics, extrinsics, feats, deps)
            # Hi = self.Hi(v, intrinsics[0], extrinsics[0], deps) # (D, 3, 3)
            # mapping_grid = self.maping_grid(Hi) # (D, H/4, W/4, 2)
            # feats_v = feats[0,v] # (32, H/4, W/4)
            # volume = self.homography_warp(feats_v, mapping_grid) # (D, 32, H/4, W/4)
            volume_sum += volume
            volume_sq_sum += volume**2

        volume_mean = volume_sum / V
        volume_var = volume_sq_sum / V - volume_mean**2
        volume_var = volume_var.unsqueeze(0)
        return volume_var

    def homography_warp1(self, idx, intrinsics, extrinsics, feats, deps):
        # idx ()
        # intrinsics (V, 3, 3)
        # extrinsics (V, 4, 4)
        # feats (B, V, C, H, W)
        # deps (D,)
        # return (D, C, H, W)
        B, V, C, H, W = feats.shape
        Hi = self.Hi(idx, intrinsics[0], extrinsics[0], deps) # (D, 3, 3)
        grid = self.maping_grid(Hi, W, H) # (D, H, W, 2)
        feats_v = feats[0,idx] # (C, H, W)
        volume = self.homography_warp(feats_v, grid) # (D, C, H, W)
        return volume

    def homography_warp(self, feats, grid):
        # feats (C, H/4, W/4)
        # grid (D, H/4, W/4, 2)
        # return (D, C, H/4, W/4)
        C, H, W = feats.shape
        D = grid.shape[0]
        feats = feats.view(C, H, W) # (C, H, W)
        grid = grid.view(D, H, W, 2) # (D, H, W, 2)
        feats = feats.unsqueeze(0).repeat(D, 1, 1, 1) # (D, C, H, W)
        # grid = grid.unsqueeze(1).repeat(1, C, 1, 1, 1) # (D, C, H, W, 2)
        # feats = feats.view(-1, H, W) # (D*C, H, W)
        grid = grid.view(-1, H, W, 2) # (D*C, H, W, 2)
        volume = torch.nn.functional.grid_sample(feats, grid, align_corners=True)
        volume = volume.view(D, C, H, W) # (D, C, H, W)
        return volume


    def Hi(self, idx, intrinsics, extrinsics, deps):
        # idx ()
        # intrinsics (V, 3, 3)
        # extrinsics (V, 4, 4)
        # deps (D,)
        D = deps.shape[0]
        R0 = extrinsics[0][:3,:3]
        T0 = extrinsics[0][:3,3]
        K0 = intrinsics[0]
        Rt0 = torch.eye(4).cuda()
        Rt0[:3,:3] = R0
        Rt0[:3,3] = T0

        Ri = extrinsics[idx][:3,:3]
        Ti = extrinsics[idx][:3,3]
        Ki = intrinsics[idx]
        Rti = torch.eye(4).cuda()
        Rti[:3,:3] = Ri
        Rti[:3,3] = Ti

        Rt0i = Rti @ Rt0.inverse()
        R = Rt0i[:3,:3]
        t = Rt0i[:3,3]
        n = torch.tensor([0, 0, -1],dtype=torch.float32).cuda()

        R = R.unsqueeze(0).repeat(D, 1, 1)
        nT = n.unsqueeze(0).repeat(D, 1, 1)
        t = t.unsqueeze(1).repeat(D, 1, 1)

        A = t @ nT
        B = A / deps[:,None,None]
        M = R - B
        Hi = Ki @ M @ K0.inverse()
        
        return Hi
    
    def maping_grid(self, Hi, img_width, img_height):
        # Hi (D, 3, 3)
        # return (D, H, W, 2)
        D = Hi.shape[0]
        camera_width = 160
        camera_height = 128
        # img_width = 160
        # img_height = 128
        
        xs = torch.linspace(0, camera_width-1, img_width)
        ys = torch.linspace(0, camera_height-1, img_height)
        y, x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([x,y,torch.ones_like(y)], axis=-1).cuda() # (H, W, 3)
        grid = grid.unsqueeze(0).repeat(D, 1, 1, 1) # (D, H, W, 3)

        flat_grid = grid.reshape(-1,1,3)
        flat_H = Hi.unsqueeze(1).repeat(1,img_width*img_height,1,1).reshape(-1,3,3)
        flat_grid = torch.bmm(flat_grid, flat_H.transpose(-1,-2))
        grid = flat_grid.view(D, img_height, img_width, 3) # (D, H, W, 3)

        grid[:,:,:,0] = grid[:,:,:,0] / grid[:,:,:,2]
        grid[:,:,:,1] = grid[:,:,:,1] / grid[:,:,:,2]
        grid = grid[:,:,:,:2]
        grid[:,:,:,0] = grid[:,:,:,0] / (camera_width - 1)
        grid[:,:,:,1] = grid[:,:,:,1] / (camera_height - 1)
        grid[:,:,:,0] = (grid[:,:,:,0] - 0.5) * 2
        grid[:,:,:,1] = (grid[:,:,:,1] - 0.5) * 2
        return grid

class CostVolumeRegularization(nn.Module):
    def __init__(self):
        super(CostVolumeRegularization, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)
    
    def forward(self, x):
        # x (B, D, C, H, W)
        B, D, C, H, W = x.shape
        x = rearrange(x, 'b d c h w -> b c d h w')
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        x = self.conv4(self.conv3(conv2))
        # x = self.conv6(self.conv5(conv4))
        # x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x) # (B, 1, D, H, W)
        return x

class DepthEstimator(nn.Module):
    def __init__(self, depth_steps=5):
        super(DepthEstimator, self).__init__()
        self.conv = nn.Conv2d(100, 100, 1)
        # self.conv1 = nn.Conv2d(depth_steps*32, 1, 1)

    def forward(self, x, deps):
        # input (B, C, D, H, W) 
        # output (B, H, W)
        
        B, C, D, H, W = x.shape
        assert B == 1, 'B must be 1 in depth estimator'
        # x = x.view(B,D*C, H, W) # (B, V*32, H, W)
        x = rearrange(x, 'b c d h w -> (b c) d h w')
        x = self.conv(x)
        x = F.softmax(x, dim=1)
        x = x.squeeze(0)
        deps = deps[:,None,None]
        x = torch.sum(x * deps, dim=0, keepdim=True)
        return x


class MVSNetWapper(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.feature_net = FeatureNet()
        self.homography_warping = HomographyWarping()
        self.cost_volume_regularization = CostVolumeRegularization()
        self.depth_estimator = DepthEstimator(depth_steps=self.cfg.depth_steps)
        self.conv1 = nn.Conv2d(3, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x: BatchedViews):
        # return (B, H, W)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16*4*4)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        intrinsics = x['intrinsics'] # (B, V, 3, 3)
        extrinsics = x['extrinsics'] # (B, V, 4, 4)
        images = x['images'] # (B, V, 3, H, W)

        # deps = torch.arange(0, self.cfg.depth_steps, self.cfg.depth_interval, device=images.device, dtype=images.dtype)
        deps = [500+i*self.cfg.depth_interval for i in range(self.cfg.depth_steps)]
        deps = torch.tensor(deps, device=images.device, dtype=images.dtype).float()
        feats = self.feature_net(images) # (B, V, 32, H/4, W/4)
        cost_volumes = self.homography_warping(intrinsics, extrinsics, feats, deps) # (B, D, 32, H/4, W/4)
        cost_volumes = self.cost_volume_regularization(cost_volumes)
        depth = self.depth_estimator(cost_volumes,deps)


        return depth

    def training_step(self, batch, batch_idx):
        intrinsics, extrinsics, imgs, depths = batch
        batched_views = BatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            images=imgs
        )
        preds = self(batched_views)
        loss = F.mse_loss(preds, depths)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        intrinsics, extrinsics, imgs, depths = batch
        batched_views = BatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            images=imgs
        )
        preds = self(batched_views)
        loss = F.mse_loss(preds, depths)
        self.log('val_loss', loss)
        if batch_idx == 0:
             # Normalize to fixed range [500, 1000] -> [0, 1]
            depths_normalized = (depths - 500) / (1000 - 500)
            preds_normalized = (preds - 500) / (1000 - 500)
            
            # Clamp to [0, 1] range
            depths_normalized = torch.clamp(depths_normalized, 0, 1).cpu()
            preds_normalized = torch.clamp(preds_normalized, 0, 1).cpu()
            
            self.logger.experiment.add_image('depth/preds', preds_normalized, self.global_step)
            self.logger.experiment.add_image('depth/groundtruth', depths_normalized, self.global_step)
            self.logger.experiment.add_image('depth/preds_colored', self.apply_colormap(preds_normalized), self.global_step)
            self.logger.experiment.add_image('depth/groundtruth_colored', self.apply_colormap(depths_normalized), self.global_step)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    print('test MVSNetWapper')
    from MVSNetDatasetDTU import MVSNetDatasetDTU
    dataset = MVSNetDatasetDTU(cfg.dataset.validation)
    print(len(dataset))
    print(dataset[0])
    exit()
    model = MVSNetWapper(cfg.model)
    print(model)

def test_homography_warp():
    print('test homography warp')

    from MVSNetDatasetDTU import MVSNetDatasetDTU, MVSNetDatasetDTUConfig
    root = './datasets/dtu'
    cfg = MVSNetDatasetDTUConfig(stage='val', root=root)
    dataset = MVSNetDatasetDTU(cfg)
    print(len(dataset))
    intrinsics, extrinsics, imgs, depth = dataset[0]
    intrinsics = intrinsics.unsqueeze(0).cuda()
    extrinsics = extrinsics.unsqueeze(0).cuda()
    imgs = imgs.unsqueeze(0).cuda()

    cost_volume = CostVolume()
    volume0 = cost_volume.homography_warp1(0, intrinsics, extrinsics, imgs, torch.tensor([500]).cuda())
    volume1 = cost_volume.homography_warp1(1, intrinsics, extrinsics, imgs, torch.tensor([500]).cuda())
    volume2 = cost_volume.homography_warp1(2, intrinsics, extrinsics, imgs, torch.tensor([500]).cuda())

    imgs_4 = imgs.squeeze(0)
    imgs_4 = torch.nn.functional.interpolate(imgs_4, size=(128, 160), mode='bilinear', align_corners=True)
    imgs_4 = imgs_4.unsqueeze(0).cuda()
    volume3 = cost_volume.homography_warp1(3, intrinsics, extrinsics, imgs_4, torch.tensor([500]).cuda())

    print(volume1.shape)


    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    axs = axs.flatten()
    for i in range(4):
        axs[i].imshow(imgs[0][i].cpu().permute(1, 2, 0))
        axs[i].set_title(f"Img {i+1}")
        axs[i].axis('off')

    axs[4].imshow(volume0[0].cpu().permute(1, 2, 0))
    axs[4].set_title(f"Volume 0")
    axs[4].axis('off')
    axs[5].imshow(volume1[0].cpu().permute(1, 2, 0))
    axs[5].set_title(f"Volume 1")
    axs[5].axis('off')
    axs[6].imshow(volume2[0].cpu().permute(1, 2, 0))
    axs[6].set_title(f"Volume 2")
    axs[6].axis('off')
    axs[7].imshow(volume3[0].cpu().permute(1, 2, 0))
    axs[7].set_title(f"Volume 3")
    axs[7].axis('off')

    # axs[8].imshow(depth.cpu())
    # axs[8].set_title(f"Depth")
    # axs[8].axis('off')
    plt.tight_layout()
    plt.show()


    # cost_volume = CostVolume()
    # intrinsics = torch.randn(1, 1, 3, 3)
    # extrinsics = torch.randn(1, 1, 4, 4)
    # feats = torch.randn(1, 1, 32, 128, 160)
    # cost_volume(intrinsics, extrinsics, feats)

if __name__ == '__main__':
    test_homography_warp()