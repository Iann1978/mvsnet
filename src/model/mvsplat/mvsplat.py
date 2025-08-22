from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn
from ..unimatch.backbone import CNNEncoder
from jaxtyping import Float, Int64
from torch import Tensor
import torch
import torch.nn.functional as F

@dataclass
class MVSPlatConfig(BaseModelConfig):
    configm: str
    feature_number: int = 128
    depth_candidates: int = 192
    depth_min: int = 425
    depth_max: int = 910

class MVSPlat(BaseModel):
    def __init__(self, cfg: MVSPlatConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.backbone = CNNEncoder(output_dim=cfg.feature_number,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 return_all_scales=False,
                 )
        self.conv1 = nn.Conv2d(cfg.feature_number*2, 1, kernel_size=3, stride=1, padding=1);
        print(self.cfg.configm)

    def forward(self, x: BatchedViews) -> Float[Tensor, "batch _ 1 _ _"]:

        # print('MVSPlat.forward:')
        # get x0, x1
        D = self.cfg.depth_candidates
        B, V, C, H, W = x['images'].shape
        # print('B, V, C, H, W:', B, V, C, H, W)
        x0 = x['images'][:,0]
        x1 = x['images'][:,1]
        assert x0.shape == (B, C, H, W), f'x0.shape: {x0.shape}'
        assert x1.shape == (B, C, H, W), f'x1.shape: {x1.shape}'

        # get features of x0, x1 through backbone(cnn)
        x0 = self.backbone(x0)[0] # [B, C, H, W]
        x1 = self.backbone(x1)[0] # [B, C, H, W]
        assert x0.shape == (B, self.cfg.feature_number, H//8, W//8), f'x0.shape: {x0.shape}'
        assert x1.shape == (B, self.cfg.feature_number, H//8, W//8), f'x1.shape: {x1.shape}'

        x = torch.cat([x0, x1], dim=1)
        x = self.conv1(x)
        assert x.shape == (B, 1, H//8, W//8), f'x.shape: {x.shape}'

        x = F.upsample(x, size=(H, W), mode='bilinear', align_corners=True)
        x = x.unsqueeze(1)
        assert x.shape == (B, 1, 1, H, W), f'x.shape: {x.shape}'

        return x