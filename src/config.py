from typing import Any, Protocol
from dataclasses import dataclass, field
import utils


@dataclass(frozen=True)
class Config:
    meta: Any
    data: Any
    model: Any
    trainer: Any


#  ╭──────────────────────────────────────────────────────────╮
#  │ Meta Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Meta:
    name: str
    id: int
    description: str
    project: str = "TEQUAL2023"
    tags: list[str] = field(default_factory=list)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Model Configurations                                     │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class ModelConfig:
    name: str
    config: Any


@dataclass
class AutoEncoderConfig:
    module: str = "models.base"
    # Set Model Architechture
    in_channels: int = 1
    img_size: int = 28 * 28
    hidden_dims: list = field(default_factory=list)
    latent_dim: int = 2


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class DataModuleConfig(Protocol):
    module: str
    data_dir: str = f"{utils.project_root_dir()}" + "data"
    num_workers: int = 0
    batch_size: int = 64
    pin_memory: bool = False


@dataclass
class MnistConfig(DataModuleConfig):
    module: str = "datasets.mnist"
    name: str = "MNIST"
    batch_size: int = 64
    in_channels: int = 1
    img_size: int = 28
    num_classes: int = 10


#  ╭──────────────────────────────────────────────────────────╮
#  │ Trainer configurations                                   │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class TrainerConfig:
    lr: float = 0.001
    num_epochs: int = 200
