"Driver for generating quotient groups"

import itertools
from multiprocessing import cpu_count

import numpy as np
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence

import utils
from topology.tequal import TEQUAL

n_jobs = cpu_count()

if __name__ == "__main__":
    params = utils.read_parameter_file()
    models = params.model_params.models
    datasets = params.data_params.datasets
    epsilon = params.clustering_params.epsilon
    combos = list(itertools.product(datasets, models))

    max_dim = 1

    for (dataset, model) in combos:
        embeddings = utils.fetch_embeddings(dataset, model)
        T = TEQUAL(data=embeddings)
        diagrams = T.quotient(epsilon=0.1)
