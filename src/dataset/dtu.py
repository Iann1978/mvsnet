from dataclasses import dataclass
from .base_dataset import BaseDataset, BaseDatasetConfig
import torch
from typing_extensions import TypedDict
from jaxtyping import Float, Int64
from torch import Tensor
from ..type.types import UnBatchedViews

@dataclass
class DTUDatasetConfig(BaseDatasetConfig):
    stage: str
    root: str



class DTUDataset(BaseDataset):
    def __init__(self, cfg: DTUDatasetConfig):
        super().__init__()
        self.cfg = cfg

    def __len__(self):
        return 100

    def __getitem__(self, idx) -> UnBatchedViews:
        V = self.cfg.view_number
        intrinsics = torch.randn(V, 3, 3)
        extrinsics = torch.randn(V, 4, 4)
        imgs = torch.randn(V, 3, 256, 256)
        targets = torch.randn(1,1,256, 256)
        # masks = torch.randn(V,1,256, 256)

        return UnBatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            imgs=imgs,
            targets=targets
        )


def shape_test():
    cfg = DTUDatasetConfig(stage='train', root='./datasets/dtu', view_number=4, type='DTU')
    dataset = DTUDataset(cfg)
    print('type of dataset[0]:', type(dataset[0]))
    print('type of dataset[0]["intrinsics"]: ', type(dataset[0]['intrinsics']), dataset[0]['intrinsics'].shape)
    print('type of dataset[0]["extrinsics"]: ', type(dataset[0]['extrinsics']), dataset[0]['extrinsics'].shape)
    print('type of dataset[0]["imgs"]: ', type(dataset[0]['imgs']), dataset[0]['imgs'].shape)
    print('type of dataset[0]["targets"]: ', type(dataset[0]['targets']), dataset[0]['targets'].shape)


if __name__ == '__main__':
    shape_test()