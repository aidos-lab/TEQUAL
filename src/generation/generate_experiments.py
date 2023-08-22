import itertools

import config
import loaders.factory as loader
import utils
from config import *
from datasets.mnist import MnistDataModule
from omegaconf import OmegaConf

#  ╭──────────────────────────────────────────────────────────╮
#  │Helper Methods                                            │
#  ╰──────────────────────────────────────────────────────────╯


def configure_datasets():
    """Initilize data sets from `params.yaml`"""
    data_params = utils.read_parameter_file()["generation_params"]["data_params"]
    datasets = data_params["datasets"]
    DataConfigs = []
    for set in datasets:
        class_name = utils.load_data_reference()[set]
        cfg = getattr(config, class_name)
        DataConfigs.append(cfg)
    return DataConfigs


def configure_models():
    """Initilize models from `params.yaml`"""
    model_params = utils.read_parameter_file()["generation_params"]["model_params"]
    modules = model_params["models"]
    hidden_dims = model_params["hidden_dims"]
    latent_dim = model_params["latent_dim"]

    # Get Number of Classes

    models = list(itertools.product(modules, hidden_dims))
    ModelConfigs = []
    for (module, hidden_dims) in models:
        # TODO: Read in a specific Config class here
        cfg = AutoEncoderConfig(
            module=module,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
        ModelConfigs.append(cfg)
    return ModelConfigs


def configure_trainers():
    trainer_params = utils.read_parameter_file()["generation_params"]["trainer_params"]
    learning_rates = trainer_params["lrs"]
    epochs = trainer_params["epochs"]

    coordinates = list(itertools.product(learning_rates, epochs))
    TrainerConfigs = []
    for lr, num_epochs in coordinates:
        trainer = TrainerConfig(lr=lr, num_epochs=num_epochs)
        TrainerConfigs.append(trainer)
    return TrainerConfigs


#  ╭──────────────────────────────────────────────────────────╮
#  │ Experiments                                              │
#  ╰──────────────────────────────────────────────────────────╯


def generate_experiments() -> None:
    """
    This experiment trains and classifies Autoencoders on the parameters
    and dataset specified in the `params.yaml` file.
    """
    params = utils.read_parameter_file()["generation_params"]
    folder = utils.create_experiment_folder()

    # Create meta data

    # Load Models from `params.yaml`
    models = configure_models()
    datasets = configure_datasets()
    trainers = configure_trainers()

    experiments = list(itertools.product(models, datasets, trainers))

    # Initialize Experiments
    for i, (model, data, trainer) in enumerate(experiments):
        meta = Meta(params["meta"], i)
        config = Config(meta, data, model, trainer)
        utils.save_config(config, folder, filename=f"config_{i}.yaml")


if __name__ == "__main__":
    generate_experiments()

    # Testing
    # cfg = OmegaConf.load(
    #     "/Users/jeremy.wayland/Desktop/projects/TEQUAL/src/generation/experiments/testing_refactor_loop/config_1.yaml"
    # )
    # dm = loader.load_module("dataset", cfg.data)
    # test_embedding = []
    # for X, _ in dm.entire_ds:
    #     test_embedding.append(X[0])
    # utils.save_embedding(test_embedding, cfg)
