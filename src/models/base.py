from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor, nn

from config import AutoEncoderConfig

from .types_ import *


class BaseVAE(nn.Module):
    def __init__(self, config: AutoEncoderConfig):
        super(BaseVAE, self).__init__()
        self.config = config

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
