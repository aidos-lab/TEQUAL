import os
import pickle
import shutil

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

#  ╭──────────────────────────────────────────────────────────╮
#  │ Utility Functions                                        │
#  ╰──────────────────────────────────────────────────────────╯


def project_root_dir():
    load_dotenv()
    root = os.getenv("root")
    return root


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


def get_experiment_dir():
    name = read_parameter_file()["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"src/experiments/{name}")
    return path


def create_experiment_folder():
    name = read_parameter_file()["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"src/experiments/{name}")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def load_data_reference():
    load_dotenv()
    root = os.getenv("root")
    YAML_PATH = os.path.join(root, f"src/datasets/reference.yaml")
    reference = OmegaConf.load(YAML_PATH)
    return reference


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_embedding(latent_representation, config):
    data_dir = os.path.join(config.data.data_dir, config.data.name)
    sub_dir = os.path.join(
        data_dir, f"embeddings/{config.meta.name}/{config.model.module}"
    )
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    file = os.path.join(sub_dir, f"embedding_{config.meta.id}")
    with open(file, "wb") as f:
        pickle.dump(latent_representation, f)


def get_embeddings_dir(dataset: str, model: str):
    root = project_root_dir()
    experiment = read_parameter_file()["experiment"]
    path = os.path.join(root, f"data/{dataset}/embeddings/{experiment}/{model}/")
    assert os.path.isdir(path), "Invalid Embeddings Directory"
    return path


def fetch_embeddings(dataset, model) -> list:
    dir = get_embeddings_dir(dataset, model)
    embeddings = []
    files = os.listdir(dir)
    files.sort()
    for file in files:
        file = os.path.join(dir, file)
        with open(file, "rb") as f:
            embedding = pickle.load(f)
        embeddings.append(embedding)
    return embeddings


def gtda_reshape(embedding):
    X = np.squeeze(embedding)
    return X.reshape(1, *X.shape)


def gtda_pad(diagrams, dims=(0, 1)):
    feature_counts = {}
    for i, diagram in enumerate(diagrams):
        feature_dims = diagram[:, 2:]
        tmp = {}
        for dim in dims:
            num_features = sum(np.where(feature_dims == dim, True, False))[0]
            tmp[dim] = num_features
        feature_counts[i] = tmp

    sizes = {}
    for dim in dims:
        counter = []
        for id in feature_counts:
            counter.append(feature_counts[id][dim])
        sizes[dim] = max(counter)

    total_features = sum(sizes.values())
    padded = np.empty(
        (
            len(diagrams),
            total_features,
            3,
        )
    )
    for i, diagram in enumerate(diagrams):
        start_idx = 0
        for dim in dims:
            stop_idx = feature_counts[i][dim]  # Current feature size
            end = sizes[dim]  # Necessary feature size

            # Input Original Diagram as subdiagram
            sub_diagram = diagram[start_idx:stop_idx]
            padded[i, start_idx:stop_idx, :] = sub_diagram

            # Insert padding
            padding = np.zeros_like(diagram[: end - stop_idx])
            # Tag correct dimension
            padding.T[2] = dim
            padded[i, stop_idx:end, :] = padding

            start_idx = end
        return padded
