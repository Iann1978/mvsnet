from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn

@dataclass
class BNetConfig(BaseModelConfig):
    configb: str

class BNet(BaseModel):
    def __init__(self, cfg: BNetConfig):
        super().__init__(cfg)
        self.backbone = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1);
        print(self.cfg.configb)

    def forward(self, x: BatchedViews):
        x = x['images'][:,0]
        x = self.backbone(x)
        return x.unsqueeze(1)