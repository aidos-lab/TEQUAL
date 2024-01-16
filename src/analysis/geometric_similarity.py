"Compare the geometric similarity of models within an equivalence class"

import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from torch.nn import PairwiseDistance

# Read in Embeddings
# Use TEQUAL + HDBSCAN to automatically generate labels
# Randomly select a coordinate in the embedding space.
# Sample from latent space from every model
# Compute average of the generated image for each cluster
# Permutation test between averages


def geometric_similarity(epsilon):
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src")
    #### SRC IMPORTS ####
    import utils
    import vis
    from topology.tequal import TEQUAL

    params = utils.read_parameter_file()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Model Training Params
    models = params.model_params.model
    datasets = params.data_params.dataset
    epsilon = params.clustering_params.epsilon
    pairs = list(itertools.product(datasets, models))

    # Topology Params
    max_dim = params.topology_params.homology_max_dim
    metric = params.clustering_params.diagram_metric
    epsilon = params.clustering_params.epsilon

    # Hyperparameter Filter
    filter_type = params.plotting_params.filter[0]
    filter_name = params.plotting_params.filter[1]
    filter_values = params[filter_type][filter_name]

    # Plotting Params
    x_axis = params.plotting_params.x_axis
    y_axis = params.plotting_params.y_axis

    # LOGGING
    log_dir = root + f"/experiments/{params.experiment}/results/similarity/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + f"similarity.log"
    logger = logging.getLogger("similarity_results")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    filter_values.sort()
    # Loop through Experiments
    logger.info(f"Geometric Similarity")
    logger.info(f"Pairs: {pairs}")
    logger.info(f"GPU?: {device}")

    embeddings, labels, all_configs = utils.fetch_embeddings(
        pairs,
        None,
        "all",  # grab all embeddings in exp
        filter_name,
    )
    embeddings = [X / utils.approximate_diameter(X) for X in embeddings]
    T = TEQUAL(data=embeddings, max_dim=max_dim)
    quotient = T.quotient(epsilon)

    configs = utils.remove_unfitted_configs(all_configs, T.dropped_point_clouds)

    logger.info(f"Landscape Distances: {T.distance_relation}")
    logger.info(f"Quotient Labels: {quotient.labels_}")
    assert len(quotient.labels_) == len(configs), "Inconsistent number of objects"

    pdist = PairwiseDistance(keepdim=True)
    matrices = {}
    embeddings = torch.tensor(np.array(embeddings)).to(device)

    for i, X in enumerate(embeddings):
        if not torch.isnan(X).any():
            matrices[i] = torch.nn.functional.pairwise_distance(
                X.unsqueeze(1), X.unsqueeze(0)
            )

    print(matrices[1].shape)
    ids = list(matrices.keys())

    # inter_cluster_norms = {}
    # intra_cluster_norms = {}

    # for label in np.unique(quotient.labels_):
    #     logger.info(msg="\n")
    #     logger.info(f"Computations for Cluster {label}")
    #     mask = np.where(quotient.labels_ == label, True, False)

    #     inter_idxs = np.array(ids)[mask]
    #     intra_idxs = np.array(ids)[~mask]

    #     logger.info(f"Staring Intercluster Comparisons")
    #     if len(inter_idxs) > 1:
    # cluster_pairs = list(itertools.combinations(inter_idxs, 2))
    # logger.info(f"Intra pairs: {cluster_pairs}")
    matrix_pairs = list(itertools.combinations(matrices, 2))
    all_correlations = []
    for i, j in matrix_pairs:
        # Loop through each sample
        logger.info(f"Comparing Distances between Embedding {i} and {j}")

        X = matrices[i]
        Y = matrices[j]

        row_correlations = []
        for v1, v2 in zip(X, Y):
            stack = torch.stack((v1, v2))
            corr = torch.corrcoef(stack)
            val = torch.min(corr)
            row_correlations.append(val)
        row_correlations = torch.tensor(row_correlations)
        all_correlations.append(torch.mean(row_correlations))
        logger.info(f"Average: {torch.mean(row_correlations)}")
        logger.info(f"Median Correlation: {torch.median(row_correlations)}")
        logger.info(f"Max Correlation: {torch.max(row_correlations)}")
        logger.info(f"Min Correlation: {torch.min(row_correlations)} ")
        logger.info("\n")

    logger.info(f"GLOBAL AVERAGE: {torch.mean(torch.tensor(all_correlations))}")


if __name__ == "__main__":
    geometric_similarity(0.001)
