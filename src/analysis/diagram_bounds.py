"Tracking stability of equivalency classes as you vary model parameters."

import itertools
import logging
import os
import pickle
import re
import sys

import numpy as np
from dotenv import load_dotenv
from gtda.diagrams import PairwiseDistance
from gtda.homology import SparseRipsPersistence as Rips
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.random_projection import GaussianRandomProjection as Gauss
from umap import UMAP

projector_reference = {
    "PCA": PCA,
    "Gaussian Random Projection": Gauss,
    # "tSNE": TSNE,
    "mMDS": MDS,
}


def extract_integer_from_filename(filename):
    # Use regular expression to extract the integer from the filename
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group())
    else:
        print("No Integer in Filename")
        return None


def read_pickled_diagrams(directory):
    pickle_files = [f for f in os.listdir(directory)]

    diagrams = {}
    for file_name in pickle_files:
        file_path = os.path.join(directory, file_name)
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            key = extract_integer_from_filename(file_name)
            if key is not None:
                diagrams[key] = data

    return diagrams


def read_matching_embedding(directory, key):
    file_name = f"embedding_{key}"
    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            return data
    else:
        print(f"File {file_name} not found in directory {directory}")
        return None


if __name__ == "__main__":
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src")
    import utils
    from topology.tequal import TEQUAL

    params = utils.read_parameter_file()

    # Unpacking Experiments
    datasets = params.data_params.dataset
    models = params.model_params.model
    pairs = list(itertools.product(datasets, models))

    # Manually Load Diagrams
    local_folder = (
        "/Users/jeremy.wayland/Desktop/ICML2024/CIFAR10-PCA-quotients-accuracy/"
    )
    diagrams_folder = local_folder + "diagrams/"

    original_diagrams = read_pickled_diagrams(diagrams_folder)

    embeddings, labels, configs = utils.fetch_embeddings(
        pairs,
        None,
        "all",  # grab all embeddings in exp
        "None",
    )
    embeddings = [
        X / utils.approximate_diameter(X, num_samples=1000) for X in embeddings
    ]

    stats = {}
    for label, projector in projector_reference.items():
        print(f"Starting Computations with {label} as Projector ")
        T = TEQUAL(data=embeddings, latent_dim=2, projector=projector)

        projected_diagrams = T.generate_diagrams()
        T.process_diagrams()
        updated_configs = [
            cfg for i, cfg in enumerate(configs) if i not in T.dropped_point_clouds
        ]
        T.quotient(epsilon=0.1)
        avg_d = np.mean(T.distance_relation)
        max_d = np.max(T.distance_relation)

        print(f"Mean Pairwise Distance :{avg_d}")
        print(f"Maximal Pairwise Distance :{max_d}")

        stability_pairs = {}
        for i, D_prime in enumerate(projected_diagrams):
            cfg_id = updated_configs[i].meta.id
            try:
                D = original_diagrams[cfg_id]
                stability_pairs[cfg_id] = [D, D_prime]
            except KeyError as e:
                print(f"No Diagram Fitted for Config {cfg_id}")

        topological_distance = PairwiseDistance(metric="landscape")
        # Compute Pairwise Distances
        scores = []
        for i, pair in stability_pairs.items():
            pair = utils.gtda_pad(pair)
            d_i = np.max(topological_distance.fit_transform(pair))
            scores.append(d_i)

        avg_loss = np.mean(scores)
        max_loss = np.max(scores)
        precision = (avg_d - max_loss * 8) / avg_d
        stats[label] = {
            "Topological Loss": np.round(max_loss, 5),
            "Average Pairwise for Projected Multiverse": np.round(avg_d, 5),
            "Clustering Bound?": bool(max_loss * 8 < max_d),
            "Precision": precision,
        }

    print(stats)
