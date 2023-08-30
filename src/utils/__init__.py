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


def copy_parameter_file(folder):
    file = os.path.join(folder, "../params.yaml")
    params = read_parameter_file()
    with open(file, "w") as f:
        OmegaConf.save(params, f)


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


def remove_duplicates(data):
    original_X = data["embedding"]
    original_labels = data["labels"]

    assert len(original_labels) == len(original_X), "MISMATCH"
    # Remove Duplicates
    clean_X, mask = np.unique(original_X, axis=0, return_index=True)
    clean_labels = original_labels[mask]

    return clean_X, clean_labels


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
            X, y = remove_duplicates(data)
        embeddings.append(X)
        labels.append(y)
    return embeddings, labels


def gtda_reshape(X):
    return X.reshape(1, *X.shape)


def gtda_pad(diagrams, dims=(0, 1)):
    homology_dims = {}
    sizes = {}
    for i, diagram in enumerate(diagrams):
        tmp = {}
        counter = {}
        for dim in dims:
            # Generate Sub Diagram for particular dim
            sub_dgm = diagram[diagram[:, :, 2] == dim]
            counter[dim] = len(sub_dgm)
            tmp[dim] = sub_dgm

        homology_dims[i] = tmp
        sizes[i] = counter

    # Building Padded Diagram Template
    total_features = 0
    template_sizes = {}
    for dim in dims:
        size = max([dgm_id[dim] for dgm_id in sizes.values()])
        template_sizes[dim] = size
        total_features += size

    template = np.zeros(
        (
            len(diagrams),
            total_features,
            3,
        )
    )
    # Populate Template
    for i in range(len(diagrams)):
        pos = 0  # position in template
        for dim in dims:
            original_len = pos + sizes[i][dim]
            template_len = pos + template_sizes[dim]
            template[i, pos:original_len, :] = homology_dims[i][dim]

            template[i, pos:template_len, 2] = int(dim)
            # Reset position for next dimension
            pos += template_sizes[dim]

    return template
