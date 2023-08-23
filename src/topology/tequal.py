import matplotlib.pyplot as plt
import numpy as np
from gtda.diagrams import PairwiseDistance
from gtda.homology import WeakAlphaPersistence
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

import utils


class TEQUAL:
    def __init__(self, data: list, max_dim: int = 1) -> None:
        self.point_clouds = [utils.gtda_reshape(X) for X in data]

        self.dims = tuple(range(max_dim + 1))

        self.alpha = WeakAlphaPersistence(homology_dimensions=self.dims)
        self.diagrams = None
        self.eq_relation = None

    def generate_diagrams(self):
        diagrams = []
        for X in self.point_clouds:
            try:
                diagram = np.squeeze(self.alpha.fit_transform(X))
                diagrams.append(diagram)
            except ValueError:
                print("TRAINING ERROR: NaNs in the latent space representation")
        self.diagrams = diagrams
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
        distance_metric = PairwiseDistance(metric=metric)

        padded_diagrams = utils.gtda_pad(self.diagrams, self.dims)
        distances = distance_metric.fit_transform(padded_diagrams)

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

    def plot_dendrogram(self, epsilon, **kwargs):
        if self.eq_relation is None:
            self.quotient(epsilon)
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
        plt.show()
        return fig
