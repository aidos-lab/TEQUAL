import networkx as nx
import numpy as np
from gtda.diagrams import PairwiseDistance
from gtda.homology import WeakAlphaPersistence
from scipy.spatial._qhull import QhullError
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import utils
from loggers.logger import Logger


class TEQUAL:
    def __init__(
        self,
        data: list,
        max_dim: int = 1,
        latent_dim: int = 2,
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
            print(f"Projecting Embeddings down to {latent_dim} Dimensions")
            self.projector = projector(n_components=latent_dim)
            data = [
                X if np.isnan(X).any() else self.projector.fit_transform(X)
                for X in data
            ]
        self.point_clouds = [utils.gtda_reshape(X) for X in data]

    def generate_diagrams(self) -> list:
        print("Generating Persistence Diagrams")
        diagrams = []
        dropped_point_clouds = []

        for i, X in enumerate(self.point_clouds):
            try:
                print(f"Diagram {i+1}/{len(self.point_clouds)}")
                diagram = self.alpha.fit_transform(X)
                diagrams.append(diagram)
                # dgm = Filtering().fit_transform(diagram)
            except (ValueError, QhullError, IndexError) as error:
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

        # Pairwise Distances
        self.compute_distances(metric=metric)

        self.eq_relation = AgglomerativeClustering(
            metric="precomputed",
            linkage=linkage,
            compute_distances=True,
            distance_threshold=epsilon,
            n_clusters=None,
        )
        self.eq_relation.fit(self.distance_relation)

        return self.eq_relation

    def compute_distances(self, metric="landscape", new=False):
        """
        Compute pairwise distances between persistence diagrams.
        :param metric: Metric to pass to gtda.PairwiseDistance.
        :param new: Recompute/overwrite existing distances.
        :return:
        """
        if new or self.distance_relation is None:
            # Pad and Clean Diagrams
            diagrams = self.process_diagrams()
            # Pairwise Distances
            self.distance_relation = PairwiseDistance(metric=metric).fit_transform(diagrams)

    # TODO untested
    def compute_set_cover(
        self,
        epsilon
    ):
        """
        Compute a set of representatives for a given set of embeddings
        such that each embedding has a representative at distance at most epsilon.
        Uses a greedy approximation to set cover that guarantees the cardinality of
        the set of representatives will be at most H(k) \in O(log k) times the size
        of the optimum.
        """

        if self.diagrams is None:
            self.generate_diagrams()

        # Log Set Cover Parameters
        self.set_cover_epsilon = epsilon

        # Compute Set Cover
        self.compute_distances()
        self.set_cover = self._compute_set_cover()

        return self.set_cover

    # TODO untested + naive implementation (but scalability is probably not an issue here)
    # TODO do we really want to set attributes _and_ return their values?
    def _compute_set_cover(self):
        """
        Compute a set-cover approximation based on a greedy bipartite-graph heuristic.
        """
        self.set_cover_representatives = dict()
        G_original = self._set_cover_graph()
        G = G_original.copy(as_view=False)
        right = {i for i in G.nodes() if i[-1] == 1}
        while right:
            rep, rep_deg = max(
                {(i, G.out_degree(i)) for i in G.nodes() if i[-1] == 0},
                key=lambda tup: tup[-1],
            )
            self.set_cover_representatives[rep[0]] = sorted(
                [i[0] for i in G_original.successors(rep)]
            )
            current_successors = list(G.successors(rep))
            G.remove_nodes_from([rep, *current_successors])
            right -= set(current_successors)
        return self.set_cover_representatives

    # TODO untested
    def _set_cover_graph(self):
        """
        Construct a bipartite graph for set-cover approximation.
        Left node set has 0 as second coordinate, right node set has 1 as second coordinate.
        There is an edge from (i,0) to (j,1) if the distance between i and j is at most epsilon.
        TODO: At most or less than?
        """
        n_probes = self.distance_relation.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from([(i, 0) for i in range(n_probes)])
        G.add_nodes_from([(i, 1) for i in range(n_probes)])
        for i in range(n_probes):
            edges = [
                ((i, 0), (j, 1))
                for j in np.argwhere(
                    self.distance_relation[i] <= self.set_cover_epsilon
                ).ravel()
            ]
            if edges:
                G.add_edges_from(edges)
        return G

    def summary(self):
        """Dendrogram height filtration"""
        pass
