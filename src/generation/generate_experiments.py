import itertools

import config
import utils
from config import *

import loaders.factory as loader


#  ╭──────────────────────────────────────────────────────────╮
#  │Helper Methods                                            │
#  ╰──────────────────────────────────────────────────────────╯


def configure_datasets():
    """Initilize data sets from `params.yaml`"""
    datasets = utils.read_parameter_file()["datasets"]
    DataConfigs = []
    for set in datasets:
        class_name = utils.load_data_reference()[set]
        cfg = getattr(config, class_name)
        DataConfigs.append(cfg)
    return DataConfigs


def configure_models():
    """Initilize models from `params.yaml`"""
    modules = utils.read_parameter_file()["models"]
    hidden_dims = utils.read_parameter_file()["hidden_dims"]

    models = list(itertools.product(modules, hidden_dims))
    ModelConfigs = []
    for (module, hidden_dims) in models:
        cfg = AutoEncoderConfig(
            module=module,
            hidden_dims=hidden_dims,
        )
        ModelConfigs.append(cfg)
    return ModelConfigs


def configure_trainers():
    learning_rates = utils.read_parameter_file()["lrs"]
    epochs = utils.read_parameter_file()["epochs"]

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
    params = utils.read_parameter_file()
    folder = utils.create_experiment_folder()

    # Create meta data
    meta = Meta(params["meta"])

    # Load Models from `params.yaml`
    models = configure_models()
    datasets = configure_datasets()
    trainers = configure_trainers()

    experiments = list(itertools.product(models, datasets, trainers))

    # Initialize Experiments
    for i, (model, data, trainer) in enumerate(experiments):
        config = Config(meta, data, model, trainer)
        utils.save_config(config, folder, filename=f"config_{i}.yaml")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    generate_experiments()
    cfg = OmegaConf.load(
        "/Users/jeremy.wayland/Desktop/projects/TEQUAL/src/generation/experiments/testing_refactor_loop/config_0.yaml"
    )
    print(loader.load_module("vanilla_vae", cfg.model))
