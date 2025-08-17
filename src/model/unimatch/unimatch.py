from dataclasses import dataclass
from model.base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn


@dataclass
class UniMatchConfig(BaseModelConfig):
    configu: str

class UniMatch(BaseModel):
    def __init__(self, cfg: UniMatchConfig):
        super().__init__(cfg)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        print(self.cfg.configu) 

    def forward(self, x: BatchedViews):
        return self.conv1(x['images'][:,0])

