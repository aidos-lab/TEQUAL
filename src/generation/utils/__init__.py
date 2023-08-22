import os
import pickle
import shutil

import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

# TODO: Which of these should be in loaders.factory?

#  ╭──────────────────────────────────────────────────────────╮
#  │ Utility Functions                                        │
#  ╰──────────────────────────────────────────────────────────╯


def save_embedding(latent_representation, config):
    data_dir = os.path.join(config.data.data_dir, config.data.name)
    sub_dir = os.path.join(data_dir, f"embeddings/{config.model.module}")
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    file = os.path.join(sub_dir, f"embedding_{config.meta.id}")
    with open(file, "wb") as f:
        pickle.dump(np.array(latent_representation), f)


def project_root_dir():
    load_dotenv()
    root = os.getenv("root")
    return root


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_experiment_path():
    name = read_parameter_file()["generation_params"]["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"src/generation/experiments/{name}")
    return path


def create_experiment_folder():
    name = read_parameter_file()["generation_params"]["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"src/generation/experiments/{name}")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def save_config(cfg, folder, filename):
    path = os.path.join(folder, filename)
    c = OmegaConf.create(cfg)
    with open(path, "w") as f:
        OmegaConf.save(c, f)


def read_parameter_file():
    load_dotenv()
    YAML_PATH = os.getenv("params")
    params = OmegaConf.load(YAML_PATH)
    return params


def load_data_reference():
    load_dotenv()
    root = os.getenv("root")
    YAML_PATH = os.path.join(root, f"src/generation/datasets/reference.yaml")
    reference = OmegaConf.load(YAML_PATH)
    return reference
