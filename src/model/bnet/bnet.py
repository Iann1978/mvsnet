from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel

@dataclass
class BNetConfig(BaseModelConfig):
    configb: str

class BNet(BaseModel):
    def __init__(self, cfg: BNetConfig):
        super().__init__(cfg)
        print(self.cfg.configb)