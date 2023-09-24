"Tracking sensitivty of (model,dataset) pairs as you tweak hyperparameters."

import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from gtda.diagrams import Amplitude, Scaler

significance = 2  # number of stds for an outlier


def scaler_fn(x):
    return np.max(x)


if __name__ == "__main__":
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src")
    #### SRC IMPORTS ####
    import utils
    from topology.tequal import TEQUAL

    params = utils.read_parameter_file()

    # Model Training Params
    models = params.model_params.model
    datasets = params.data_params.dataset
    epsilon = params.clustering_params.epsilon
    pairs = list(itertools.product(datasets, models))

    # Topology Params
    max_dim = params.topology_params.homology_max_dim
    metric = params.clustering_params.diagram_metric

    # Hyperparameter Filter
    filter_type = params.plotting_params.filter[0]
    filter_name = params.plotting_params.filter[1]
    filter_values = params[filter_type][filter_name]

    # LOGGING
    log_dir = root + f"/experiments/{params.experiment}/results/anomaly_detection/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + f"anomaly_detection_{filter_name}.log"
    logger = logging.getLogger("anomaly_detection_results")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    filter_values.sort()
    # Loop through Experiments
    logger.info(f"Pairs: {pairs}")
    for key_val in filter_values:
        embeddings, labels, configs = utils.fetch_embeddings(
            pairs,
            key_val,
            filter_type,
            filter_name,
        )
        T = TEQUAL(data=embeddings, max_dim=max_dim)
        T.generate_diagrams()
        diagrams = T.process_diagrams()

        scaler = Scaler(
            metric=metric,
            function=scaler_fn,
        )
        scaled_diagrams = scaler.fit_transform(diagrams)

        # Landscape Norms
        A = Amplitude(metric=metric, order=2)
        norms = A.fit_transform(scaled_diagrams)
        sigma = np.std(norms)
        mu = np.mean(norms)

        anomalous_cfgs = []
        anomalous_embeddings = []
        for i, norm in enumerate(norms):
            if abs(norm - mu) > significance * sigma:
                anomalous_cfgs.append(configs[i])
                anomalous_embeddings.append(embeddings[i])

        for i, X in enumerate(anomalous_embeddings):
            plt.scatter(X.T[0], X.T[1], label=anomalous_cfgs[i].meta.id, alpha=0.5)

        plt.legend()
        plt.show()
        logger.info(f"Key: {key_val} | Anomalous Configs: {anomalous_cfgs}")
        logger.info(f"Mu: {mu}")
        logger.info(f"Sigma: {sigma}")
        logger.info(f"Significance: {significance}")
        logger.info("\n")
