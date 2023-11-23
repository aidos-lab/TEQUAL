import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from config import DataModuleConfig
from datasets.base_dataset import DataModule
from loaders.factory import register


class FashionMNISTDataModule(DataModule):
    """
    FashionMNIST
    """

    def __init__(self, config: DataModuleConfig):
        super().__init__(config)
        self.config = config

    def setup(self):
        """
        This method constructs the entire dataset, note we concatenate the train/test datasets.
        This allows for k-fold cross-validation later on.
        If you prefer to create the splits yourself, set the variables
        self.train_ds, self.test_ds, and self.val_ds.
        """
        entire_dataset = torch.utils.data.ConcatDataset(
            [
                FashionMNIST(
                    root=self.config.data_dir,
                    train=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5], std=[0.5]),
                        ]
                    ),
                    download=True,
                ),
                FashionMNIST(
                    root=self.config.data_dir,
                    train=False,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5], std=[0.5]),
                        ]
                    ),
                    download=True,
                ),
            ]
        )
        return entire_dataset


def initialize():
    register("dataset", FashionMNISTDataModule)
