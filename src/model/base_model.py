from dataclasses import dataclass
from abc import abstractmethod, ABC
import torch.nn as nn
from typing_extensions import TypedDict
from jaxtyping import Float, Int64
from torch import Tensor
from ..type.types import BatchedViews
from pytorch_lightning import LightningModule

@dataclass
class BaseModelConfig:
    type: str



class BaseModel(nn.Module, ABC):
    def __init__(self, cfg: BaseModelConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, x: BatchedViews):
        pass

