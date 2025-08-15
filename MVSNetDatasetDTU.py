from torch.utils.data import Dataset
from typing import TypedDict, Literal, List
from jaxtyping import Float, Int64
from torch import Tensor
import os
import torch
import numpy as np
import imageio.v3 as imageio
from omegaconf import DictConfig
from dataclasses import dataclass


Stage = Literal["train", "val", "test"]

@dataclass
class MVSNetDatasetDTUConfig:
    stage: Stage
    root: str
    src_view_number: int
   


class Meta(TypedDict, total=True):
    scan: str
    light: int
    views: List[int]



class MVSNetDatasetDTU(Dataset):
    metas: List[Meta]
    viewpairs: List[List[int]]

    def __init__(self, cfg: MVSNetDatasetDTUConfig):
        self.cfg = cfg
        self.view_pairs_filepath = os.path.join('configs', 'dtu_meta', 'view_pairs.txt')
        self.view_pairs = self.read_view_pairs(self.view_pairs_filepath)
        self.scan_filepath = os.path.join('configs', 'dtu_meta', f'{cfg.stage}_all.txt')
        self.scans = self.read_scans(self.scan_filepath)
        self.metas = self.build_metas(self.scans)
    
    def build_metas(self, scans):
        metas = []
        for scan in scans:
            for light in range(7):
                for view_pair in self.view_pairs:
                    view_pair = view_pair[:self.cfg.src_view_number+1]
                    metas.append({"scan": scan, "light": light, "views": view_pair})
        return metas

    def read_scans(self, filename):
        with open(filename) as f:
            scans = [line.rstrip() for line in f.readlines()]
        return scans

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

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
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


        return intrinsics, extrinsics, imgs, depth, mask


def basic_test():
    print('test MVSNetDatasetDTU')

    root = './datasets/dtu'
    cfg = MVSNetDatasetDTUConfig(stage='val', root=root, src_view_number=4)
    dataset = MVSNetDatasetDTU(cfg)
    print(len(dataset))
    intrinsics, extrinsics, imgs, depth, mask = dataset[0]
    print('intrinsics:', intrinsics.shape)
    print('extrinsics:', extrinsics.shape)
    print('imgs:', imgs.shape, imgs.min(), imgs.max())
    print('depth:', depth.shape, depth.min(), depth.max())
    print('mask:', mask.shape, mask.min(), mask.max())

def show_basic_test():
    print('show basic test')
    root = './datasets/dtu'
    cfg = MVSNetDatasetDTUConfig(stage='val', root=root, src_view_number=4)
    dataset = MVSNetDatasetDTU(cfg)
    print(len(dataset))
    intrinsics, extrinsics, imgs, depth, mask = dataset[0]
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    view_number = cfg.src_view_number + 1 if cfg.src_view_number < 6 else 7
    for i in range(view_number):
        axs[i].imshow(imgs[i].cpu().permute(1, 2, 0))
        axs[i].set_title(f"Img {i+1}")
        axs[i].axis('off')
    axs[view_number].imshow(depth.cpu())
    axs[view_number].set_title(f"Depth")
    axs[view_number].axis('off')
    axs[view_number+1].imshow(mask.cpu())
    axs[view_number+1].set_title(f"Mask")
    axs[view_number+1].axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    show_basic_test()