from omegaconf import DictConfig
from dacite import from_dict
from .anet.anet import ANetConfig, ANet
from .bnet.bnet import BNetConfig, BNet
from .unimatch.unimatch import UniMatchConfig, UniMatch

config_factory = {'ANet': ANetConfig, 'BNet': BNetConfig, 'UniMatch': UniMatchConfig}
model_factory = {'ANet': ANet, 'BNet': BNet, 'UniMatch': UniMatch}
def get_model(model_config: DictConfig):
    print(model_config)
    config_type = model_config['type']
    typed_model_config = from_dict(config_factory[config_type], model_config)
    model = model_factory[config_type](typed_model_config)
    return model