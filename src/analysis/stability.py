"Tracking stability of equivalency classes as you vary model parameters."

import itertools
import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src")
    import utils

    params = utils.read_parameter_file()

    # Unpacking Experiments
    datasets = params.data_params.dataset
    models = params.model_params.model

    matrices = {}
    filter_type = params.plotting_params.filter[0]
    filter_name = params.plotting_params.filter[1]
    filter_values = params[filter_type][filter_name]

    # LOGGING
    log_dir = root + f"/experiments/{params.experiment}/results/stability/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + f"stability_{filter_name}.log"
    logger = logging.getLogger("stability_results")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Loop through Experiments
    for key_val in filter_values:
        distances = utils.load_distance_matrix(filter_name, key_val)
        if len(distances) == 5:
            matrices[key_val] = distances

    matrix_pairs = list(itertools.combinations(matrices, 2))

    for x, y in matrix_pairs:
        logger.info(f"({x,y})")
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
        logger.info(f"Average over all rows: {np.mean(row_correlations)}")
        logger.info(f"Median Correlation: {np.median(row_correlations)}")
        logger.info(f"Max Correlation: {np.max(row_correlations)}")
        logger.info(f"Min Correlation: {np.min(row_correlations)} ")
        logger.info("\n")
