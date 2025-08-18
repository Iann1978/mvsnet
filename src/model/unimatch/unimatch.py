from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel, BatchedViews
import torch.nn as nn
from .backbone import CNNEncoder

@dataclass
class UniMatchConfig(BaseModelConfig):
    configu: str

class UniMatch(BaseModel):
    def __init__(self, cfg: UniMatchConfig):
        super().__init__(cfg)
        self.backbone = CNNEncoder(output_dim=128,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 return_all_scales=True,
                 )
        self.conv1 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        print(self.cfg.configu) 

    def forward(self, x: BatchedViews):
        print(x['images'].shape)
        x = self.backbone(x['images'][:,0])
        print('\n\n\n')
        print(len(x))
        for i in range(len(x)):
            print(x[i].shape)
        exit()
        return self.conv1(x[0]).unsqueeze(1)

