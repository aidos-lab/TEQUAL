import torch
import torchvision.transforms as transforms
from torchvision.datasets import LFWPeople as LFW

from config import DataModuleConfig
from datasets.base_dataset import DataModule
from loaders.factory import register


class LFWPeopleDataModule(DataModule):
    """
    LFW People
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
                LFW(
                    root=self.config.data_dir,
                    split="train",
                    transform=transforms.Compose(
                        [
                            transforms.Resize((224, 224)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=15),
                            transforms.ColorJitter(
                                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                            ),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                            ),
                        ]
                    ),
                    download=True,
                ),
                LFW(
                    root=self.config.data_dir,
                    split="test",
                    transform=transforms.Compose(
                        [
                            transforms.Resize((224, 224)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=15),
                            transforms.ColorJitter(
                                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                            ),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                            ),
                        ]
                    ),
                    download=True,
                ),
            ]
        )
        return entire_dataset


def initialize():
    register("dataset", LFWPeopleDataModule)
