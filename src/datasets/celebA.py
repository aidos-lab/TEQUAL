import torch
import torchvision.transforms as transforms
from torchvision.datasets import CelebA

from config import DataModuleConfig
from datasets.base_dataset import DataModule
from loaders.factory import register


class CelebADataModule(DataModule):
    """
    CelebA
    """

    def __init__(self, config: DataModuleConfig):
        super().__init__(config)
        self.config = config

    def setup(self):
        """
        This method constructs the entire dataset, note we concatenate the train/test datasets.
        This allows for k-fold cross validation later on.
        If you prefer to create the splits yourself, set the variables
        self.train_ds, self.test_ds and self.val_ds.
        """
        entire_dataset = torch.utils.data.ConcatDataset(
            [
                CelebA(
                    root=self.config.data_dir,
                    split="train",
                    transform=transforms.Compose(
                        [
                            transforms.Resize(size=(64, 64)),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                ),
                CelebA(
                    root=self.config.data_dir,
                    split="test",
                    transform=transforms.Compose(
                        [
                            transforms.Resize(size=(64, 64)),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                ),
            ]
        )
        return entire_dataset


def initialize():
    register("dataset", CelebADataModule)
