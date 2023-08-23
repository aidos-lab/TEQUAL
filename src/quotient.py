"Driver for generating quotient groups"

import itertools

import numpy as np

import utils
from topology.tequal import TEQUAL

if __name__ == "__main__":

    params = utils.read_parameter_file()
    models = params.generation_params.model_params.models
    datasets = params.generation_params.data_params.datasets

    combos = list(itertools.product(datasets, models))

    for (dataset, model) in combos:
        embeddings = utils.fetch_embeddings(dataset, model)
        T = TEQUAL(data=embeddings)
        print(T.distances)
