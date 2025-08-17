from dataclasses import dataclass
from model.base_model import BaseModelConfig, BaseModel
from model.anet.a1net import A1NetConfig, A1Net
from typing import Any
from omegaconf import DictConfig
from dacite import from_dict

config_factory = {'A1Net': A1NetConfig}
model_factory = {'A1Net': A1Net}
def get_model(model_config: DictConfig):
    print(model_config)
    config_type = model_config['type']
    typed_model_config = from_dict(config_factory[config_type], model_config)
    model = model_factory[config_type](typed_model_config)
    return model

@dataclass
class ANetConfig(BaseModelConfig):
    configa: str
    subnet: Any

class ANet(BaseModel):
    def __init__(self, cfg: ANetConfig):
        super().__init__(cfg)
        self.subnet = get_model(self.cfg.subnet)
        print(self.cfg.configa)