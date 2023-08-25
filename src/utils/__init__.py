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
    path = os.path.join(root, f"src/experiments/{name}/configs/")
    return path


def create_experiment_folder():
    name = read_parameter_file()["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"src/experiments/{name}/configs/")
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
    labels = []
    files = os.listdir(dir)
    files.sort()
    for file in files:
        file = os.path.join(dir, file)
        with open(file, "rb") as f:
            data = pickle.load(f)
        embeddings.append(data["embedding"])
        labels.append(data["labels"])
    return embeddings, labels


def gtda_reshape(embedding):
    X = np.squeeze(np.array(embedding))
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
    new_diagrams = np.empty(
        (
            len(diagrams),
            total_features,
            3,
        )
    )

    for i, diagram in enumerate(diagrams):

        start = 0
        for dim in dims:
            sub_length = start + feature_counts[i][dim]
            pad_length = start + sizes[dim]
            sub = diagram[start:sub_length, :]
            new_diagrams[i, start:sub_length, :] = sub

            padding = np.zeros(
                shape=(1, pad_length - sub_length, 3),
            )
            padding[:, :, 2:] = int(dim)
            new_diagrams[i, sub_length:pad_length, :] = padding
            start = pad_length
    return new_diagrams
