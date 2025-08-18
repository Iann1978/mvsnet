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
class RNDDatasetConfig(BaseDatasetConfig):
    config_rnd: str

class RNDDataset(BaseDataset):
    def __init__(self, cfg: RNDDatasetConfig, stage: str):
        super().__init__()
        self.cfg = cfg
        self.stage = stage

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        V = self.cfg.view_number
        intrinsics = torch.randn(V, 3, 3)
        extrinsics = torch.randn(V, 4, 4)
        imgs = torch.randn(V, 3, 256, 256)
        targets = torch.randn(1, 256, 256)
        return UnBatchedViews(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            imgs=imgs,
            targets=targets)