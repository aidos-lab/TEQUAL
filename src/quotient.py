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
    name = params.experiment

    # Model Training Params
    models = params.model_params.model
    datasets = params.data_params.dataset
    epsilon = params.clustering_params.epsilon
    pairs = list(itertools.product(datasets, models))

    # Topology Params
    max_dim = params.topology_params.homology_max_dim

    # Plotting Params
    # TODO: add plotting params fn
    filter_type = params.plotting_params.filter[0]
    filter_name = params.plotting_params.filter[1]
    filter_values = params[filter_type][filter_name]

    x_axis = params.plotting_params.x_axis

    plots = []
    for key_val in filter_values:
        embeddings, labels, configs = utils.fetch_embeddings(
            pairs,
            key_val,
            filter_name="sample_size",
            filter_type="data",
        )
        T = TEQUAL(data=embeddings, max_dim=max_dim)

        T.generate_diagrams()

        T.quotient(epsilon)

        embedding_grid = vis.visualize_embeddings(
            T,
            key_val,
            filter_name,
            labels=labels,
            x_axis=params.plotting_params.x_axis,
            y_axis=params.plotting_params.y_axis,
        )
        dendrogram, colormap = vis.visualize_dendrogram(T)
        experiment = utils.get_experiment_dir()

        out_dir = os.path.join(experiment, f"../results/")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        file = os.path.join(out_dir, f"{name}_{filter_name}_{key_val}.html")
        plots.append(embedding_grid)
        plots.append(dendrogram)
        vis.save_visualizations_as_html([embedding_grid, dendrogram], file)

        # Only produce one diagram
        if filter_name == "all":
            break
    file = os.path.join(out_dir, f"{name}_summary.html")
    vis.save_visualizations_as_html(plots, file)
