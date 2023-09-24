"Compare the similarity of models within an equivalence class"

import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from sklearn.metrics import pairwise_distances
from torchvision import transforms


def centeroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    C = torch.Tensor([sum_x / length, sum_y / length])
    C = C.view(1, *C.shape)
    return C


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


if __name__ == "__main__":
    load_dotenv()
    root = os.getenv("root")
    sys.path.append(root + "src")
    #### SRC IMPORTS ####
    import utils
    import vis
    from topology.tequal import TEQUAL

    params = utils.read_parameter_file()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Model Training Params
    models = params.model_params.model
    datasets = params.data_params.dataset
    epsilon = params.clustering_params.epsilon
    pairs = list(itertools.product(datasets, models))

    # Topology Params
    max_dim = params.topology_params.homology_max_dim
    metric = params.clustering_params.diagram_metric
    epsilon = params.clustering_params.epsilon

    # Hyperparameter Filter
    filter_type = params.plotting_params.filter[0]
    filter_name = params.plotting_params.filter[1]
    filter_values = params[filter_type][filter_name]

    # Plotting Params
    x_axis = params.plotting_params.x_axis
    y_axis = params.plotting_params.y_axis

    # LOGGING
    log_dir = root + f"/experiments/{params.experiment}/results/centroid_comparisons/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + f"centroid.log"
    logger = logging.getLogger("centroid_comparison")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    filter_values.sort()
    # Loop through Experiments
    logger.info(f"Pairs: {pairs}")

    embeddings, labels, all_configs = utils.fetch_embeddings(
        pairs,
        None,
        "all",  # grab all embeddings in exp
        filter_name,
    )

    T = TEQUAL(data=embeddings, max_dim=max_dim)
    quotient = T.quotient(epsilon)

    configs = utils.remove_unfitted_configs(all_configs, T.dropped_point_clouds)

    logger.info(f"configs: {configs}")
    logger.info(f"Labels: {quotient.labels_}")
    assert len(quotient.labels_) == len(configs), "Inconsistent number of objects"

    # Invert Image Normalization
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    generated_imgs = {}
    for i, cfg in enumerate(configs):
        C = centeroid(embeddings[i])
        C = C.to(device)
        model_id = cfg.meta.id
        model = utils.load_model(model_id)
        model.to(device)
        img = model.decode(C).detach().cpu()

        # img = unnormalize(img)
        generated_imgs[model_id] = img.view(64, 64, 3).numpy()

    pairs_imgs = itertools.combinations(generated_imgs.keys(), 2)
    norms = []
    for i, j in pairs_imgs:
        logger.info(f"Distance Between Centroids for Models: {(i,j)}")
        img_i = generated_imgs[i]
        img_j = generated_imgs[j]

        norm = np.linalg.norm(img_i - img_j)
        norms.append(norm)
        logger.info(f"{norm}")

    print(np.mean(norms))
    print(np.std(norms))
