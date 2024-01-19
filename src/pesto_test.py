import numpy as np

from topology.pesto import PESTO

if __name__ == "__main__":
    metric = PESTO(max_homology_dim=2, normalize=True)

    X = np.random.random(size=(20000, 10))
    Y = np.random.random(size=(20000, 8))

    score = metric.fit_transform(X, Y, N=15)

    print(score)
