import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from gtda.diagrams import Filtering, PairwiseDistance, Scaler
from gtda.homology import WeakAlphaPersistence
from scipy.spatial._qhull import QhullError
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from tqdm import tqdm

import utils
from loggers.logger import Logger


class TEQUAL:
    def __init__(
        self,
        data: list,
        max_dim: int = 1,
        latent_dim: int = 3,
        max_edge_length=5,
        projector=PCA,
        n_cpus: int = 4,
    ) -> None:
        self.dims = tuple(range(max_dim + 1))
        self.filtration_inf = max_edge_length
        self.alpha = WeakAlphaPersistence(
            homology_dimensions=self.dims,
            max_edge_length=max_edge_length,
            n_jobs=n_cpus,
        )
        self.diagrams = None
        self.distance_relation = None
        self.eq_relation = None
        self.logger = Logger

        # Preprocessing Data
        if latent_dim > 0:
            self.projector = projector(n_components=latent_dim)
            data = [self.projector.fit_transform(X) for X in data]
        self.point_clouds = [utils.gtda_reshape(X) for X in data]
        print(self.point_clouds[0].shape)

    def generate_diagrams(self) -> list:
        diagrams = []
        dropped_point_clouds = []

        for i, X in enumerate(self.point_clouds):
            try:
                print(f"Diagram {i+1}/{len(self.point_clouds)}")
                diagram = self.alpha.fit_transform(X)
                diagrams.append(diagram)
                # dgm = Filtering().fit_transform(diagram)
            except (ValueError, QhullError) as error:
                self.logger.log(
                    f"TRAINING ERROR: {error} NaNs in the latent space representation"
                )
                # Track Dropped Diagrams
                dropped_point_clouds.append(i)

        self.diagrams = diagrams
        self.dropped_point_clouds = dropped_point_clouds
        return self.diagrams

    def process_diagrams(self):
        # Pad Diagrams
        padded_diagrams = utils.gtda_pad(self.diagrams, self.dims)

        # Remove inf features -> replace with large value
        # np.nan_to_num(padded_diagrams, posinf=posinf, copy=False)

        # TODO: Implement Scaler
        # scaler = Scaler(metric=metric)
        # self.scaled_diagrams = scaler.fit_transform(self.diagrams)
        return padded_diagrams

    def quotient(
        self,
        epsilon,
        metric: str = "landscape",
        linkage: str = "average",
    ) -> AgglomerativeClustering:
        if self.diagrams is None:
            self.generate_diagrams()

        # Log Quotient Parameters
        self.epsilon = epsilon
        self.metric = metric
        self.linkage = linkage

        # Pad and Clean Diagrams
        diagrams = self.process_diagrams()

        # Pairwise Distances
        distance_metric = PairwiseDistance(metric="landscape")
        self.distance_relation = distance_metric.fit_transform(diagrams)

        self.eq_relation = AgglomerativeClustering(
            metric="precomputed",
            linkage=linkage,
            compute_distances=True,
            distance_threshold=epsilon,
            n_clusters=None,
        )
        self.eq_relation.fit(self.distance_relation)

        return self.eq_relation

    def summary(self):
        """Dendrogram height filtration"""
        pass
