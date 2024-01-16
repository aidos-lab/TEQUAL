"Driver for generating quotient groups"

import itertools
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import pandas as pd
from gtda.homology import WeakAlphaPersistence
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from sklearn.manifold import MDS

import utils
import vis
from topology.tequal import TEQUAL

n_jobs = cpu_count()


def get_colorbrewer_color(value, N):
    color_map = get_cmap("viridis")
    return color_map(value / N)


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

        T = TEQUAL(data=embeddings, max_dim=max_dim, latent_dim=2)
        T.generate_diagrams()
        T.quotient(epsilon)

        distances = T.distance_relation

        print(distances.shape)

        print(f"Num Dropped Point Clouds: {len(T.dropped_point_clouds)}")

        updated_configs = [
            cfg for i, cfg in enumerate(configs) if i not in T.dropped_point_clouds
        ]
        # utils.save_distance_matrix(distances, filter_name, key_val)
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
        # dendrogram, colormap = vis.visualize_dendrogram(T, updated_configs)
        # print(colormap)

        # fig, ax = plt.subplots(figsize=(10, 10))
        # for i, X in enumerate(embeddings):
        #     ax.scatter(X.T[0], X.T[1], alpha=0.5, label=configs[i].meta.id)

        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.xaxis.set_tick_params(labelbottom=False)
        # ax.yaxis.set_tick_params(labelleft=False)

        # ax.set_xticks([])
        # ax.set_yticks([])
        # # ax.axis("off")
        # # ax.legend()
        # plt.show()

        fig2, ax2 = plt.subplots(figsize=(10, 10))

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_embedding = mds.fit_transform(distances)

        # Plot the MDS embedding
        cfg_ids = [int(cfg.meta.id) for cfg in updated_configs]
        x = mds_embedding[:, 0]
        y = mds_embedding[:, 1]
        for i, x_point, y_point in zip(cfg_ids, x, y):
            ax2.scatter(
                x_point,
                y_point,
                label=f"Config {i}",
                s=20,
                c=[get_colorbrewer_color(i, N=64)],
            )
        ax2.set_title(
            f"{params.experiment}: 2D MDS Embedding of Hyperameter Multiverse"
        )
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")
        # ax2.legend(
        #     fontsize="small",
        #     loc="upper right",
        #     bbox_to_anchor=(1.1, 1),
        # )

        plt.show()
