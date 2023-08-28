import matplotlib.pyplot as plt
import numpy as np
from gtda.diagrams import Filtering, PairwiseDistance, Scaler
from gtda.homology import WeakAlphaPersistence
from sklearn.cluster import AgglomerativeClustering

import utils
from loggers.logger import Logger


class TEQUAL:
    def __init__(self, data: list, labels: list, max_dim: int = 1) -> None:

        self.point_clouds = [utils.gtda_reshape(X) for X in data]
        self.labels = labels

        self.dims = tuple(range(max_dim + 1))

        self.alpha = WeakAlphaPersistence(homology_dimensions=self.dims)
        self.diagrams = None
        self.distance_relation = None
        self.eq_relation = None
        self.logger = Logger

    def generate_diagrams(self) -> list:
        diagrams = []
        for X in self.point_clouds:
            try:
                diagram = self.alpha.fit_transform(X)
                # dgm = Filtering().fit_transform(diagram)
                diagrams.append(diagram)
            except (ValueError) as error:
                self.logger.log(
                    f"TRAINING ERROR: {error} NaNs in the latent space representation"
                )
        self.diagrams = diagrams
        return self.diagrams

    def process_diagrams(self, metric: str = "landscape"):
        # Pad Diagrams
        padded_diagrams = utils.gtda_pad(self.diagrams, self.dims)
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

        diagrams = self.process_diagrams(metric)
        # Pairwise Distances
        distance_metric = PairwiseDistance(metric=metric)

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
