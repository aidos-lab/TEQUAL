"Driver for generating quotient groups"

import itertools
import os
from multiprocessing import cpu_count

import utils
import vis
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
        diagrams = T.generate_diagrams()
        for diagram in diagrams:
            print(len(diagrams))
        T.quotient(epsilon)

        embedding_grid = vis.visualize_embeddings(T)
        dendrogram, colormap = vis.visualize_dendrogram(T)

        experiment = utils.get_experiment_dir()
        out_dir = os.path.join(experiment, f"../results/{dataset}/")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        file = os.path.join(out_dir, f"{model}.html")

        vis.save_visualizations_as_html([embedding_grid, dendrogram], file)
