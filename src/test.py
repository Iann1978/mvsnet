from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import hydra
from dacite import from_dict
from typing import Any
from model import get_model




@dataclass
class TestConfig:
    name: str
    model: Any




@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    print(cfg.model.type)

    cfg = from_dict(TestConfig, OmegaConf.to_container(cfg))
    print(cfg)
    print(cfg.model)
    print(cfg.model['type'])

    model = get_model(cfg.model)
    print(model)
    print(model.cfg)

    # config_type = cfg.model['type']
    # model_config = from_dict(config_factory[config_type], cfg.model)
    # print(model_config)

    # model = model_factory[config_type](model_config)
    # print(model)


if __name__ == '__main__':
    main()