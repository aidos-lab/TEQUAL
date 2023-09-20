"Tracking sensitivty of (model,dataset) pairs as you tweak hyperparameters."

import itertools
import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv
from gtda.diagrams import Amplitude, Scaler


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
    log_dir = root + f"/experiments/{params.experiment}/results/sensitivity/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + f"sensitivity_{filter_name}.log"
    logger = logging.getLogger("sensitivity_results")
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
        score = np.std(norms)
        logger.info(f"{filter_name}: {key_val}")
        logger.info(f"Score: {score}")
        logger.info("\n")
