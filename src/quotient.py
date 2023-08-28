"Driver for generating quotient groups"

import itertools
import os
from multiprocessing import cpu_count

from gtda.homology import WeakAlphaPersistence

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

    max_dim = params.topology_params.homology_max_dim

    for (dataset, model) in combos:
        embeddings, labels = utils.fetch_embeddings(dataset, model)
        T = TEQUAL(data=embeddings, max_dim=max_dim)

        T.generate_diagrams()

        T.quotient(epsilon)
        embedding_grid = vis.visualize_embeddings(T, labels)
        dendrogram, colormap = vis.visualize_dendrogram(T)
        experiment = utils.get_experiment_dir()
        out_dir = os.path.join(experiment, f"../results/{dataset}/")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        file = os.path.join(out_dir, f"{model}.html")
        vis.save_visualizations_as_html([embedding_grid, dendrogram], file)
