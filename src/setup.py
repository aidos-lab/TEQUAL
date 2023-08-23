import itertools

import config
import utils
from config import *

#  ╭──────────────────────────────────────────────────────────╮
#  │Helper Methods                                            │
#  ╰──────────────────────────────────────────────────────────╯


def configure_datasets():
    """Initilize data sets from `params.yaml`"""
    data_params = utils.read_parameter_file()["data_params"]
    datasets = data_params["datasets"]
    DataConfigs = []
    for set in datasets:
        class_name = utils.load_data_reference()[set]
        cfg = getattr(config, class_name)
        DataConfigs.append(cfg)
    return DataConfigs


def configure_models():
    """Initilize models from `params.yaml`"""
    model_params = utils.read_parameter_file()["model_params"]
    modules = model_params["models"]
    hidden_dims = model_params["hidden_dims"]
    latent_dims = model_params["latent_dims"]

    # Model Space
    models = list(itertools.product(modules, hidden_dims, latent_dims))
    ModelConfigs = []
    # Enumerate Model Configs
    for (module, hidden_dims, latent_dim) in models:
        cfg = AutoEncoderConfig(
            module=module,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
        ModelConfigs.append(cfg)
    return ModelConfigs


def configure_trainers():
    trainer_params = utils.read_parameter_file()["trainer_params"]
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
    params = utils.read_parameter_file()
    folder = utils.create_experiment_folder()

    # Create meta data

    # Load Models from `params.yaml`
    models = configure_models()
    datasets = configure_datasets()
    trainers = configure_trainers()

    experiments = list(itertools.product(models, datasets, trainers))

    # Initialize Experiments
    for i, (model, data, trainer) in enumerate(experiments):
        meta = Meta(
            name=params["experiment"],
            id=i,
            description=params["description"],
        )
        config = Config(meta, data, model, trainer)
        utils.save_config(config, folder, filename=f"config_{i}.yaml")


if __name__ == "__main__":
    generate_experiments()
