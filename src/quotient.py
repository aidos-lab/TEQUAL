"Driver for generating quotient groups"

import itertools
import os
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
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

    # Hyperparameter Filter
    filter_type = params.plotting_params.filter[0]
    filter_name = params.plotting_params.filter[1]
    filter_values = params[filter_type][filter_name]

    # Plotting Params
    # TODO: add plotting params fn
    x_axis = params.plotting_params.x_axis
    y_axis = params.plotting_params.y_axis

    plots = []
    for key_val in filter_values:
        embeddings, labels, configs = utils.fetch_embeddings(
            pairs,
            key_val,
            filter_type,
            filter_name,
        )
        T = TEQUAL(data=embeddings, max_dim=max_dim)
        T.generate_diagrams()
        T.quotient(epsilon)

        distances = T.distance_relation

        print(distances.shape)

        print(len(T.dropped_point_clouds))

        utils.save_distance_matrix(distances, filter_name, key_val)
        # embedding_grid = vis.visualize_embeddings(
        #     T,
        #     key_val,
        #     filter_name,
        #     configs,
        #     labels=labels,
        #     x_axis=x_axis,
        #     y_axis=y_axis,
        # )

        # embedding_grid.show()
        # embedding_grid.write_image(f"./{name}_grid.svg")
        dendrogram, colormap = vis.visualize_dendrogram(T, configs)
        # dendrogram.write_image(f"./{name}_clustering.svg")
        dendrogram.show()
        fig, ax = plt.subplots(figsize=(10, 10))
        for i, X in enumerate(embeddings):
            ax.scatter(X.T[0], X.T[1], alpha=0.5, label=configs[i].meta.id)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis("off")
        # ax.legend()
        plt.show()

        # fig.savefig(f"./{name}_embeddings.png")

        # dendrogram.show()
        # dendrogram.write_image(f"./{name}_dendrogram.svg")
        # experiment = utils.get_experiment_dir()

    #     out_dir = os.path.join(experiment, f"../results/plots/")
    #     if not os.path.isdir(out_dir):
    #         os.makedirs(out_dir, exist_ok=True)
    #     file = os.path.join(out_dir, f"{name}_{filter_name}_{key_val}.html")
    #     plots.append(embedding_grid)
    #     plots.append(dendrogram)
    #     vis.save_visualizations_as_html([embedding_grid, dendrogram], file)

    #     # Only produce one diagram
    #     if filter_name == "all":
    #         break
    # file = os.path.join(out_dir, f"{name}_summary.html")
    # vis.save_visualizations_as_html(plots, file)
