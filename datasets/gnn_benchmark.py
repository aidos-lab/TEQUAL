from datasets.base_dataset import DataModule, DataModuleConfig

import torch
from torch_geometric.datasets import GNNBenchmarkDataset
import torchvision.transforms as transforms
from torch_geometric.data import Data

from loaders.factory import register
from dataclasses import dataclass
from base_dataset import DataModuleConfig


@dataclass
class GNNBenchmarkDataModuleConfig(DataModuleConfig):
    name: str = "MNIST"
    module: str = "datasets.gnn_benchmark"


#  ╭──────────────────────────────────────────────────────────╮
#  │ Transforms                                               │
#  ╰──────────────────────────────────────────────────────────╯


class ThresholdTransform(object):
    def __call__(self, data):
        x = torch.hstack([data.pos, data.x])
        new_data = Data(x=x, edge_index=data.edge_index, y=data.y)
        return new_data


class CenterTransform(object):
    def __call__(self, data):
        data.x -= data.x.mean()
        data.x /= data.x.pow(2).sum(axis=1).sqrt().max()
        return data


#  ╭──────────────────────────────────────────────────────────╮
#  │ Datasets                                                 │
#  ╰──────────────────────────────────────────────────────────╯


class GNNBenchmarkDataModule(DataModule):
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([ThresholdTransform(), CenterTransform()])
        super().__init__(config.root, config.batch_size, config.num_workers)

    def prepare_data(self):
        GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="train",
        )
        GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="test",
        )
        GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="val",
        )

    def setup(self):
        self.train_ds = GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="train",
        )
        self.test_ds = GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="test",
        )
        self.val_ds = GNNBenchmarkDataset(
            name=self.config.name,
            root=self.config.root,
            pre_transform=self.transform,
            split="val",
        )


def initialize():
    register("dataset", GNNBenchmarkDataModule)
