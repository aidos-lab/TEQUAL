"Config Files for Datasets, Models, Trainers, etc."

from dataclasses import dataclass, field
from typing import Any, Protocol

import utils


@dataclass(frozen=True)
class Config:
    meta: Any
    data_params: Any
    model_params: Any
    trainer_params: Any


#  ╭──────────────────────────────────────────────────────────╮
#  │ Meta Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Meta:
    name: str
    id: int
    description: str
    project: str = "Topological Quotients"
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
    hidden_dims: list = field(default_factory=list)
    kernel_size: int = 3
    alpha: float = None
    latent_dim: int = 2


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


# Defaults
@dataclass
class DataModuleConfig(Protocol):
    module: str
    data_dir: str = f"{utils.project_root_dir()}" + "data/"
    num_workers: int = 4
    batch_size: int = 64
    pin_memory: bool = False
    sample_size: float = None


@dataclass
class XycConfig(DataModuleConfig):
    module: str = "datasets.xyc"
    name: str = "XYC"
    in_channels: int = 1
    img_size: int = 84
    num_classes: int = 3


@dataclass
class MnistConfig(DataModuleConfig):
    module: str = "datasets.mnist"
    name: str = "MNIST"
    in_channels: int = 1
    img_size: int = 28
    num_classes: int = 10


@dataclass
class CIFAR10Config(DataModuleConfig):
    module: str = "datasets.cifar10"
    name: str = "CIFAR-10"
    in_channels: int = 3
    img_size: int = 32
    num_classes: int = 10


@dataclass
class CelebAConfig(DataModuleConfig):
    module: str = "datasets.celebA"
    name: str = "celebA"
    in_channels: int = 3
    img_size: int = 64
    num_classes: int = 40


@dataclass
class LFWPeopleConfig(DataModuleConfig):
    module: str = "datasets.lfw_people"
    name: str = "LFWPeople"
    in_channels: int = 3
    img_size: int = 200


@dataclass
class FashionMNISTConfig(DataModuleConfig):
    module: str = "datasets.fashion_mnist"
    name: str = "FashionMNIST"
    in_channels: int = 1
    img_size: int = 28


#  ╭──────────────────────────────────────────────────────────╮
#  │ Trainer configurations                                   │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class TrainerConfig:
    lr: float = 0.001
    num_epochs: int = 200
