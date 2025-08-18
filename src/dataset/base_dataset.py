from dataclasses import dataclass
from typing import List, Dict, Any, abstractmethod
from torch.utils.data import Dataset
from ..type.types import UnBatchedViews
from abc import ABC

@dataclass
class BaseDatasetConfig:
    type: str
    view_number: int

class BaseDataset(Dataset, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> UnBatchedViews:
        pass

