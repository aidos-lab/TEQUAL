import numpy as np
import ripser
from gtda.diagrams import PairwiseDistance
from gtda.homology import WeakAlphaPersistence
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

import utils


class TEQUAL:
    def __init__(self, data: list, normalize: bool = False, max_dim: int = 0) -> None:
        self.point_clouds = [np.squeeze(X) for X in data]
        self.distances = np.array([pairwise_distances(X) for X in self.point_clouds])
        self.thresholds = np.array([D.max() for D in self.distances])
        if normalize:
            self.point_clouds = [
                X / self.thresholds[i] for i, X in enumerate(self.point_clouds)
            ]

        self.max_dim = max_dim

        self.rips = ripser.Rips(maxdim=self.max_dim, verbose=False)
        self.alpha = WeakAlphaPersistence(homology_dimensions=self.max_dim)
        self.diagrams = None
        self.eq_relation = None

    def generate_diagrams(self):
        self.diagrams = [self.alpha.fit_transform(X) for X in self.point_clouds[:1]]
        # self.diagrams = utils.convert_to_gtda(dgms, max_dim=self.max_dim)
        return self.diagrams

    def quotient(
        self,
        epsilon,
        metric: str = "landscape",
        linkage: str = "average",
    ) -> AgglomerativeClustering:

        if self.diagrams is None:
            self.generate_diagrams()

        # Pairwise Distances
        distance_metric = PairwiseDistance(metric=metric, order=1)
        distances = distance_metric.transform(self.diagrams)

        self.eq_relation = AgglomerativeClustering(
            metric="precomputed",
            linkage=linkage,
            compute_distances=True,
            distance_threshold=epsilon,
            n_clusters=None,
        )
        self.eq_relation.fit(distances)

        return self.eq_relation

    def summary(self):
        """Dendrogram height filtration"""
        pass

    def plot_dendrogram(self, **kwargs):
        if self.eq_relation is None:
            self.quotient()
        counts = np.zeros(self.eq_relation.children_.shape[0])
        n_samples = len(self.eq_relation.labels_)
        for i, merge in enumerate(self.eq_relation.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [self.eq_relation.children_, self.eq_relation.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        fig = dendrogram(linkage_matrix, **kwargs)
        return fig
