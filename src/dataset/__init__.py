from omegaconf import DictConfig
from dacite import from_dict
from .dtu import DTUDatasetConfig, DTUDataset

config_factory = {'DTU': DTUDatasetConfig}
dataset_factory = {'DTU': DTUDataset}
def get_dataset(dataset_config: DictConfig, stage: str):
    # print(dataset_config)
    config_type = dataset_config['type']
    typed_dataset_config = from_dict(config_factory[config_type], dataset_config)
    dataset = dataset_factory[config_type](typed_dataset_config, stage)
    return dataset