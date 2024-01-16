import os
import pickle
import re
import shutil
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist

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


def sort_configs(cfg_name):
    id = [int(i) for i in re.findall(r"\d+", cfg_name)]
    return id


def load_config(id, folder):
    path = os.path.join(folder, f"config_{id}.yaml")
    cfg = OmegaConf.load(path)
    return cfg


def read_parameter_file():
    load_dotenv()
    YAML_PATH = os.getenv("params")
    params = read_parameter_file_from_path(YAML_PATH)
    return params

def read_parameter_file_from_path(YAML_PATH):
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
    path = os.path.join(root, f"experiments/{name}/configs/")
    return path


def get_models_dir():
    name = read_parameter_file()["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"experiments/{name}/models/")
    return path


def create_experiment_folder():
    name = read_parameter_file()["experiment"]
    root = project_root_dir()
    path = os.path.join(root, f"experiments/{name}/configs/")
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
    data_dir = os.path.join(config.data_params.data_dir, config.data_params.name)
    sub_dir = os.path.join(
        data_dir, f"embeddings/{config.meta.name}/{config.model_params.module}"
    )
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    file = os.path.join(sub_dir, f"embedding_{config.meta.id}")
    with open(file, "wb") as f:
        pickle.dump(latent_representation, f)


def save_model(model, id):
    path = get_models_dir()

    if not os.path.isdir(path):
        os.makedirs(path)

    file = os.path.join(path, f"model_{id}")
    torch.save(model, f=file)


def save_distance_matrix(distances, filter_name, filter_val):
    root = project_root_dir()
    params = read_parameter_file()
    out_dir = os.path.join(root, f"experiments/{params.experiment}/results/distances/")
    #if not os.path.isdir(out_dir): # You don't need to check this, exist_ok=True takes care of that
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"distances_{filter_name}_{filter_val}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(distances, f)


def load_distance_matrix(filter_type, filter_val):
    params = read_parameter_file()
    distances = load_distance_matrix_for_experiment(params, filter_type, filter_val)
    return distances


def load_distance_matrix_for_experiment(params, filter_type, filter_val):
    root = project_root_dir()
    distances_in_file = os.path.join(
        root,
        "experiments/"
        + params.experiment
        + "/results/distances/"
        + f"distances_{filter_type}_{filter_val}.pkl",
    )
    with open(distances_in_file, "rb") as D:
        distances = pickle.load(D)
    return distances


def load_model(id):
    folder = get_models_dir()
    file = os.path.join(folder, f"model_{id}")
    model = torch.load(file)
    return model


def get_embeddings_dir(dataset: str, model: str):
    root = project_root_dir()
    experiment = read_parameter_file().experiment
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


def fetch_embeddings(
    pairs, key_val, filter_type="data_params", filter_name="sample_size"
) -> list:
    embeddings = []
    exp = get_experiment_dir()

    for dataset, model in pairs:
        dir = get_embeddings_dir(dataset, model)
        files = os.listdir(dir)
        for file in files:
            file = os.path.join(dir, file)
            # Config_ID
            id = int(re.search(r"\d+$", file).group())
            config = load_config(id, exp)
            if filter_name == "model":
                filter_name = "module"
            if filter_type == "all":
                with open(file, "rb") as f:
                    data = pickle.load(f)
                    X, y = remove_duplicates(data)
                embeddings.append((X, y, id))
            elif config[filter_type][filter_name] == key_val:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                    X, y = remove_duplicates(data)
                embeddings.append((X, y, id))

    # Sort by Config_ID
    embeddings = sorted(
        embeddings,
        key=lambda x: x[-1],
    )

    # Unpack
    points = [itemgetter(0)(item) for item in embeddings]
    labels = [itemgetter(1)(item) for item in embeddings]
    configs = [load_config(itemgetter(2)(item), exp) for item in embeddings]

    return points, labels, configs


def gtda_reshape(X):
    X = pd.DataFrame(data=X).dropna(axis=0).values
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


def remove_unfitted_configs(configs, idxs):
    # print(f"Removing {len(idxs)} config file(s)!")
    new_configs = configs.copy()
    for i in idxs:
        elem = configs[i]
        new_configs.remove(elem)
    return new_configs


def approximate_diameter(points, num_samples=10000):
    "Approximate Diameter"
    subset = [np.random.choice(len(points))]
    for _ in range(num_samples - 1):
        distances = cdist([points[subset[-1]]], points).ravel()
        new_point = np.argmax(distances)
        subset.append(new_point)
    pairwise_distances = cdist(points[subset], points[subset])
    return np.max(pairwise_distances)
