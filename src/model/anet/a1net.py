from dataclasses import dataclass
from ..base_model import BaseModelConfig, BaseModel

@dataclass
class A1NetConfig(BaseModelConfig):
    configa1: str

class A1Net(BaseModel):
    def __init__(self, cfg: A1NetConfig):
        super().__init__(cfg)
        print(self.cfg.configa1)