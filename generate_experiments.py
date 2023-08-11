import os
from omegaconf import OmegaConf
import shutil
from datasets.tu import TUDataModuleConfig
from models.base_model import ModelConfig

from config import Config, TrainerConfig, Meta
from models.custom_model import NNConfig
from datasets.custom_dataset import MnistConfig

"""
This script creates all the configurations in the config folder. 
It allows for better reproducibility. The script takes 
NOTE: Gets ran every time make run is called.
"""


#  ╭──────────────────────────────────────────────────────────╮
#  │ Helper methods                                           │
#  ╰──────────────────────────────────────────────────────────╯


def create_experiment_folder(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def save_config(cfg, path):
    c = OmegaConf.create(cfg)
    with open(path, "w") as f:
        OmegaConf.save(c, f)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Experiments                                              │
#  ╰──────────────────────────────────────────────────────────╯


def custom_example() -> None:
    """
    This experiment trains and classifies the letter high dataset in
    the TU dataset.
    """

    experiment = "./experiment/custom_example"
    create_experiment_folder(experiment)

    # Create meta data
    meta = Meta("desct-test-new")

    # Create Trainer Config
    trainer = TrainerConfig(lr=0.0001, num_epochs=10)

    # Create NN model config
    model = NNConfig()

    # Create the dataset config.
    data = MnistConfig()

    config = Config(meta, data, model, trainer)
    save_config(config, os.path.join(experiment, f"config.yaml"))


if __name__ == "__main__":
    custom_example()
