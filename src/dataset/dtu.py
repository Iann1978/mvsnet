from dataclasses import dataclass
from .base_dataset import BaseDataset, BaseDatasetConfig
import torch
from typing_extensions import TypedDict
from jaxtyping import Float, Int64
from torch import Tensor
from ..type.types import UnBatchedViews
import os
from typing import List
import imageio.v3 as imageio
import numpy as np

@dataclass
class DTUDatasetConfig(BaseDatasetConfig):
    root: str

class Meta(TypedDict, total=True):
    scan: str
    light: int
    views: List[int]

class DTUDataset(BaseDataset):
    metas: List[Meta]
    viewpairs: List[List[int]]

    def __init__(self, cfg: DTUDatasetConfig, stage: str):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_pairs_filepath = os.path.join('configs', 'dtu_meta', 'view_pairs.txt')
        self.view_pairs = self.read_view_pairs(self.view_pairs_filepath)
        self.scan_filepath = os.path.join('configs', 'dtu_meta', f'{self.stage}_all.txt')
        self.scans = self.read_scans(self.scan_filepath)
        self.metas = self.build_metas(self.scans)


    def build_metas(self, scans):
        metas = []
        for scan in scans:
            for light in range(7):
                for view_pair in self.view_pairs:
                    view_pair = view_pair[:self.cfg.view_number]
                    metas.append({"scan": scan, "light": light, "views": view_pair})
        return metas
    def read_view_pairs(self, filename) -> List[List[int]]:
        view_pairs = []
        with open(self.view_pairs_filepath) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                view_pairs.append([ref_view] + src_views)
        return view_pairs

    def read_scans(self, filename):
        with open(filename) as f:
            scans = [line.rstrip() for line in f.readlines()]
        return scans

    def read_image(self, filename):
        img = imageio.imread(filename)
        return img

    def read_depth(self, filename):
        depth = imageio.imread(filename)
        depth = np.flipud(depth).copy()
        depth = torch.from_numpy(depth).float()
        return depth

    def read_mask(self, filename):
        mask = imageio.imread(filename)
        mask = torch.from_numpy(mask).float()
        mask = mask / 255.0
        return mask
    
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsic = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsic = extrinsic.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsic = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsic = intrinsic.reshape((3, 3))
        # depth_min & depth_interval: line 11
        # depth_min = float(lines[11].split()[0]) * self.scale_factor
        # depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        # near_far = [depth_min, depth_max]
        return intrinsic, extrinsic

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx) -> UnBatchedViews:
        """
        return:
            intrinsics: (V, 3, 3)
            extrinsics: (V, 4, 4)
            imgs: (V, 3, H, W)
            depth: (H, W)
        """
        meta = self.metas[idx]
        scan = meta['scan']
        light = meta['light']
        views = meta['views']

        intrinsics = []
        extrinsics = []
        imgs = []
        depths = []
        for view in views:
            view_pathfile = os.path.join(self.cfg.root, 'Cameras', f'train/{view:08d}_cam.txt')
            img_pathfile = os.path.join(self.cfg.root, 'Rectified', f'{scan}_train', f'rect_{view+1:03d}_{light:01d}_r5000.png')

            intrinsic, extrinsic = self.read_cam_file(view_pathfile)
            img = self.read_image(img_pathfile)
            
            intrinsics.append(torch.from_numpy(intrinsic).float())
            extrinsics.append(torch.from_numpy(extrinsic).float())
            imgs.append(torch.from_numpy(img).float())


        intrinsics = torch.stack(intrinsics)
        extrinsics = torch.stack(extrinsics)
        imgs = torch.stack(imgs).permute (0, 3, 1, 2)/255.0

        refview = views[0]
        depth_pathfile = os.path.join(self.cfg.root, 'Depths', f'{scan}_train', f'depth_map_{refview:04d}.pfm')
        mask_pathfile = os.path.join(self.cfg.root, 'Depths', f'{scan}_train', f'depth_visual_{refview:04d}.png')
        depth = self.read_depth(depth_pathfile)
        mask = self.read_mask(mask_pathfile)

        # 4x upsample depth and mask
        import torch.nn.functional as F
        depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), scale_factor=4, mode='nearest')

        return UnBatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            imgs=imgs,
            targets=depth,
            masks=mask
        )


def shape_test():
    cfg = DTUDatasetConfig(stage='val', root='./datasets/dtu', view_number=4, type='DTU')
    dataset = DTUDataset(cfg)
    print('type of dataset[0]:', type(dataset[0]))
    print('type of dataset[0]["intrinsics"]: ', type(dataset[0]['intrinsics']), dataset[0]['intrinsics'].shape)
    print('type of dataset[0]["extrinsics"]: ', type(dataset[0]['extrinsics']), dataset[0]['extrinsics'].shape)
    print('type of dataset[0]["imgs"]: ', type(dataset[0]['imgs']), dataset[0]['imgs'].shape)
    print('type of dataset[0]["targets"]: ', type(dataset[0]['targets']), dataset[0]['targets'].shape)

def show_part():
    view_number = 4
    cfg = DTUDatasetConfig(stage='val', root='./datasets/dtu', view_number=view_number, type='DTU')
    dataset = DTUDataset(cfg)
    unbatched_views = dataset[0]
    intrinsics = unbatched_views['intrinsics']
    extrinsics = unbatched_views['extrinsics']
    imgs = unbatched_views['imgs']
    depth = unbatched_views['targets']

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    for i in range(view_number):
        axs[i].imshow(imgs[i].cpu().permute(1, 2, 0))
        axs[i].set_title(f"Img {i+1}")
        axs[i].axis('off')
    axs[view_number].imshow(depth[0].cpu().permute(1, 2, 0))
    axs[view_number].set_title(f"Depth")
    axs[view_number].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_part()