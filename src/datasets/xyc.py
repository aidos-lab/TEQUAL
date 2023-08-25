import os
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import DataModuleConfig
from datasets.base_dataset import DataModule
from loaders.factory import register


class XYC(Dataset):
    def __init__(self, file):
        self.file = file
        self.data_zip = np.load(self.file)
        self.data = self.data_zip["imgs"]
        self.labels = self.data_zip["labs"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        label = self.labels[idx]

        sample = torch.from_numpy(np.expand_dims(sample, axis=0))
        label = torch.from_numpy(np.expand_dims(label, axis=0))

        sample = (sample, label)

        return sample


class XycDataModule(DataModule):
    def __init__(self, config: DataModuleConfig):
        super().__init__(config)
        self.config = config

    def circle(self, img, center, radius, color):
        m, n = img.shape
        center = np.array(center, dtype=np.float32)
        x = np.arange(0, m)

        coords = product(x, x)
        coords = np.array(list(coords), dtype=np.float32)

        in_circle = np.where(np.linalg.norm(coords - center, axis=-1) < radius)[0]
        img[
            coords[in_circle].astype(np.uint8)[:, 0],
            coords[in_circle].astype(np.uint8)[:, 1],
        ] = color

        return img

    def generate(self):
        datasets = []
        labels = []

        for i in tqdm(range(15, 84 - 15 - 1)):
            for j in range(15, 84 - 15 - 1):
                for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    template = np.zeros((84, 84), dtype=np.float32)
                    datasets.append(self.circle(template, (j, i), 15, c))
                    labels.append(np.array([j, i, c]))

        datasets = np.stack(datasets)
        labels = np.stack(labels)

        self.dataset_folder_name = os.path.join(
            self.config.data_dir, f"{self.config.name}/raw"
        )

        if not os.path.isdir(self.dataset_folder_name):
            os.makedirs(self.dataset_folder_name)

        np.savez(
            self.file,
            imgs=datasets,
            labs=labels,
        )

    def setup(self):
        # Check for existing data
        self.dataset_folder_name = os.path.join(
            self.config.data_dir, f"{self.config.name}/raw/"
        )
        self.file = os.path.join(self.dataset_folder_name, "xyc.npz")
        if not os.path.isfile(self.file):
            self.generate()
        self.train_ds = XYC(self.file)
        self.val_ds = XYC(self.file)
        self.test_ds = XYC(self.file)

        entire_ds = torch.utils.data.ConcatDataset(
            [self.train_ds, self.test_ds, self.val_ds]
        )
        return entire_ds


def initialize():
    register("dataset", XycDataModule)
