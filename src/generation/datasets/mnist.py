from datasets.base_dataset import DataModule
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from loaders.factory import register
from config import DataModuleConfig


class MnistDataModule(DataModule):
    """
    Example implementation for an existing dataset.
    Note that we do the transform here, that is why we
    need to create a separate class for the "new"
    dataset.
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
                MNIST(
                    root=self.config.data_dir,
                    train=True,
                    transform=transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            # transforms.Lambda(lambda x: torch.flatten(x)),
                        ]
                    ),
                    download=True,
                ),
                MNIST(
                    root=self.config.data_dir,
                    train=False,
                    transform=transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            # transforms.Lambda(lambda x: torch.flatten(x)),
                        ]
                    ),
                    download=True,
                ),
            ]
        )
        return entire_dataset


def initialize():
    register("dataset", MnistDataModule)
