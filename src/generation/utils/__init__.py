import os
from omegaconf import OmegaConf
import shutil
from dotenv import load_dotenv

# TODO: Which of these should be in loaders.factory?

#  ╭──────────────────────────────────────────────────────────╮
#  │ Utility Functions                                        │
#  ╰──────────────────────────────────────────────────────────╯


def project_root_dir():
    load_dotenv()
    root = os.getenv("root")
    return root


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_experiment_path():
    name = read_parameter_file()["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"src/generation/experiments/{name}")
    return path


def create_experiment_folder():
    name = read_parameter_file()["experiment"]
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
