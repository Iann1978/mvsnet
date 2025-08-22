from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn

@dataclass
class MVSPlatConfig(BaseModelConfig):
    configm: str

class MVSPlat(BaseModel):
    def __init__(self, cfg: MVSPlatConfig):
        super().__init__(cfg)
        self.backbone = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1);
        print(self.cfg.configm)

    def forward(self, x: BatchedViews):
        x = x['images'][:,0]
        x = self.backbone(x)
        return x.unsqueeze(1)