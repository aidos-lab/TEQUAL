import os
import pickle
import shutil

import numpy as np
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
    name = read_parameter_file()["generation_params"]["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"src/experiments/{name}")
    return path


def create_experiment_folder():
    name = read_parameter_file()["generation_params"]["experiment"]
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
    experiment = read_parameter_file()["generation_params"]["experiment"]
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


def convert_to_gtda(diagrams, max_dim):
    homology_dimensions = range(max_dim + 1)

    slices = {
        dim: slice(None) if (dim) else slice(None, -1) for dim in homology_dimensions
    }
    Xt = [
        {dim: diagram[dim][slices[dim]] for dim in homology_dimensions}
        for diagram in diagrams
    ]
    start_idx_per_dim = np.cumsum(
        [0]
        + [
            np.max([len(diagram[dim]) for diagram in Xt] + [1])
            for dim in homology_dimensions
        ]
    )
    min_values = [
        min(
            [
                np.min(diagram[dim][:, 0]) if diagram[dim].size else np.inf
                for diagram in Xt
            ]
        )
        for dim in homology_dimensions
    ]
    min_values = [min_value if min_value != np.inf else 0 for min_value in min_values]
    n_features = start_idx_per_dim[-1]
    Xt_padded = np.empty((len(Xt), n_features, 3), dtype=float)

    for i, dim in enumerate(homology_dimensions):
        start_idx, end_idx = start_idx_per_dim[i : i + 2]
        padding_value = min_values[i]
        # Add dimension as the third elements of each (b, d) tuple globally
        Xt_padded[:, start_idx:end_idx, 2] = dim
        for j, diagram in enumerate(Xt):
            subdiagram = diagram[dim]
            end_idx_nontrivial = start_idx + len(subdiagram)
            # Populate nontrivial part of the subdiagram
            if len(subdiagram) > 0:
                Xt_padded[j, start_idx:end_idx_nontrivial, :2] = subdiagram
            # Insert padding triples
            Xt_padded[j, end_idx_nontrivial:end_idx, :2] = [padding_value] * 2

    return Xt_padded
