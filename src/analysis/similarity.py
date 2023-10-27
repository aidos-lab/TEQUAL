"Compare the similarity of models within an equivalence class"

import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import pairwise_distances

# Read in Embeddings
# Use TEQUAL + HDBSCAN to automatically generate labels
# Randomly select a coordinate in the embedding space.
# Sample from latent space from every model
# Compute average of the generated image for each cluster
# Permutation test between averages


if __name__ == "__main__":
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
    logger.info(f"Pairs: {pairs}")

    embeddings, labels, all_configs = utils.fetch_embeddings(
        pairs,
        None,
        "all",  # grab all embeddings in exp
        filter_name,
    )

    T = TEQUAL(data=embeddings, max_dim=max_dim)
    quotient = T.quotient(epsilon)

    configs = utils.remove_unfitted_configs(all_configs, T.dropped_point_clouds)

    logger.info(f"configs: {configs}")
    logger.info(f"Labels: {quotient.labels_}")
    assert len(quotient.labels_) == len(configs), "Inconsistent number of objects"

    samples = torch.randn(10, params.model_params.latent_dim[0]).to(device)

    imgs = {}
    for label in np.unique(quotient.labels_):

        mask = np.where(quotient.labels_ == label, True, False)
        idxs = np.array(range(len(configs)))[mask]
        for i in idxs:
            cfg = configs[i]
            model_id = cfg.meta.id
            model = utils.load_model(model_id)
            model.to(device)
            imgs[model_id] = {label: model.decode(samples).detach().cpu().numpy()}

    for model in imgs:

        imgs[model]

    cpu_samples = samples.detach().cpu().numpy()

    # embedding_grid = vis.visualize_embeddings(
    #     T,
    #     None,
    #     filter_name,
    #     configs,
    #     labels=labels,
    #     x_axis=x_axis,
    #     y_axis=y_axis,
    # )

    # dendrogram, colormap = vis.visualize_dendrogram(T, configs)

    # embedding_grid.show()
    # dendrogram.show()

    matrices = {}
    for i, X in enumerate(embeddings):
        # Pairwise distances between embeddings
        matrices[i] = pairwise_distances(X, metric="euclidean")

    matrix_pairs = list(itertools.combinations(matrices, 2))

    for x, y in matrix_pairs:
        logger.info(f"Computing correlation for {(x,y)}")
        row_correlations = []
        X = matrices[x]
        Y = matrices[y]

        for i in range(len(X)):
            result = np.corrcoef(
                X[i],
                Y[i],
            )
            val = np.min(result)
            row_correlations.append(val)
        logger.info(f"Average: {np.mean(row_correlations)}")
        logger.info(f"Median Correlation: {np.median(row_correlations)}")
        logger.info(f"Max Correlation: {np.max(row_correlations)}")
        logger.info(f"Min Correlation: {np.min(row_correlations)} ")
        logger.info("\n")

    plt.scatter(
        X.T[0],
        X.T[1],
        alpha=0.3,
        label=f"Embedding {i}",
    )
    plt.scatter(cpu_samples.T[0], cpu_samples.T[1], c="black", label="Samples")
    plt.legend()
    # plt.show()
