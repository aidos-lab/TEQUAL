"Driver for generating quotient groups"

import itertools
from multiprocessing import cpu_count
import pickle

from gtda.homology import WeakAlphaPersistence

import utils
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

    # Re: Filters in `fetch_embeddings()`– useful for visualizing 2D slices of a grid search.
    # Leaving these out for now, so you load all embeddings in a grid search.

    # Get All Embeddings
    embeddings, labels, configs = utils.fetch_embeddings(
        pairs,
        key_val=None,
        filter_type="all",
        filter_name=params.plotting_params.filter[1],
    )

    # UNNORMALIZED
    embeddings = [
        X  for X in embeddings
    ]
    T = TEQUAL(
        data=embeddings, max_dim=1, latent_dim=2
    )  # latent dim here is output dimension of PCA
    # TODO calling the line below generates weird [Errno 9] Bad file descriptor errors
    # When the logger is set up and whenever it tries to log something
    # Probably not important, but might confuse people trying to work with pregenerated results
    T.generate_diagrams()

    # TODO FIX we currently only record the excluded diagrams in self.dropped_point_clouds
    # This makes it hard to reconstruct which points belong to which embeddings

    filter_name = "landscape"
    filter_val = "unnormalized"
    # TODO distances do not depend on filter_val, so naming convention for dist matrices is confusing
    T.compute_distances(metric=filter_name)
    print()
    print(f"Unnormalized Multiverse Metric Space of {params.experiment}")
    utils.save_distance_matrix(T.distance_relation, filter_name, filter_val)

    # NORMALIZED
    embeddings = [
        X / utils.approximate_diameter(X, num_samples=1000) for X in embeddings
    ]
    T = TEQUAL(
        data=embeddings, max_dim=1, latent_dim=2
    )  # latent dim here is output dimension of PCA
    T.generate_diagrams()
    filter_name = "landscape"
    filter_val = "normalized"
    T.compute_distances(metric=filter_name, new=True)
    print()
    print(f"Normalized Multiverse Metric Space of {params.experiment}")
    utils.save_distance_matrix(T.distance_relation, filter_name, filter_val)



