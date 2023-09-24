import argparse
import sys
import time

import numpy as np
import torch
from codecarbon import EmissionsTracker
from omegaconf import OmegaConf

import loaders.factory as loader
import utils
import wandb
from loggers.logger import Logger, timing
from metrics.metrics import compute_recon_loss

torch._C._mps_emptyCache()

my_experiment = utils.read_parameter_file().experiment
mylogger = Logger(exp=my_experiment, name="training_results", dev=False)


class Experiment:
    def __init__(self, experiment, logger, dev=True):
        """
        Creates the setup and does inits
        - Loads datamodules
        - Loads models
        - Initializes logger
        """

        self.config = OmegaConf.load(experiment)
        self.dev = dev
        self.logger = logger
        self.device = torch.device(
            "mps:0" if torch.backends.mps.is_available() else "cpu"
        )

        # Load the dataset
        self.dm = loader.load_module("dataset", self.config.data_params)

        # Set model input size
        self.config.model_params.img_size = self.config.data_params.img_size
        self.config.model_params.in_channels = self.config.data_params.in_channels

        # Load the model
        model = loader.load_module("model", self.config.model_params)
        # Send model to device
        self.model = model.to(self.device)
        # Loss function and optimizer.
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.trainer_params.lr
        )
        # WANDB Config
        self.config.meta.tags = [
            self.config.model_params.module,
            self.config.data_params.module,
            f"Sample Size: {self.config.data_params.sample_size}",
        ]
        self.logger.wandb_init(self.model, self.config)

    @timing(mylogger)
    def run(self):
        """
        Runs an experiment given the loaded config files.
        """
        tot_co2_emission = 0
        start = time.time()
        for epoch in range(self.config.trainer_params.num_epochs):
            stats, c02_emission = self.run_epoch()

            tot_co2_emission += c02_emission
            reported_loss = self.loss.item()

            self.logger.log(
                msg=f"epoch {epoch} | train loss {reported_loss:.2f}",
                params={
                    "train loss": reported_loss,
                    "CO2 emission (in Kg)": c02_emission,
                },
            )
            if "Reconstruction_Loss" in stats.keys():
                recon_loss = stats["Reconstruction_Loss"]
                self.logger.log(
                    msg=f"epoch {epoch} | train recon loss {recon_loss:.2f}"
                )

            if epoch % 10 == 0:
                end = time.time()
                self.compute_metrics(epoch)
                self.logger.log(
                    msg=f"Training the model 10 epochs took: {end - start:.2f} seconds."
                )

                start = time.time()

        self.finalize_run()

    def run_epoch(self):
        with EmissionsTracker(
            gpu_ids="0",
            tracking_mode="process",
        ) as tracker:
            tracker.start()
            loader = self.dm.train_dataloader()
            for batch_idx, (x, y) in enumerate(loader):
                # Convert to float32 for MPS
                x, y = x.float(), y.float()
                X, _ = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                results = self.model(X)

                stats = self.model.loss_function(
                    *results,
                    batch_idx=batch_idx,
                    M_N=0.00025,
                    optimizer_idx=0,
                )
                self.loss = stats["loss"]
                self.loss.backward()

                self.optimizer.step()

            # get co2 emissions from tracker
            emissions = tracker.stop()
            return stats, emissions

        # Delete Train Loader
        del loader

    @timing(mylogger)
    def finalize_run(self):
        # Train Reconstruction Loss
        test_loss = compute_recon_loss(self.model, self.dm.test_dataloader())
        epoch = self.config.trainer_params.num_epochs
        # Log statements
        self.logger.log(
            f"epoch {epoch} | test recon loss {test_loss:.2f}",
            params={
                "test_loss": test_loss,
            },
        )

    @timing(mylogger)
    def compute_metrics(self, epoch):

        val_loss = compute_recon_loss(self.model, self.dm.val_dataloader())

        # Log statements to console
        self.logger.log(
            msg=f"epoch {epoch} | val loss { val_loss.item():.2f}",
            params={"epoch": epoch, "val_loss": val_loss.item()},
        )

    @timing(mylogger)
    def save_run(self):

        embedding = torch.Tensor()
        labels = torch.Tensor()
        if np.isnan(self.loss.item()):
            self.logger.log(msg="Not saving embedding: Nans in loss function")
        else:
            loader = self.dm.full_dataloader()
            for x, y in loader:
                train_data, train_labels = x.float().to(self.device), y.float()
                labels = torch.cat((labels, train_labels))

                # Run Model on Batch and send to CPU
                batch_embedding = self.model.latent(train_data).detach().cpu()

                # Update Embedding
                embedding = torch.cat((embedding, batch_embedding))

            # Send to CPU
            embedding_cpu = embedding.numpy()
            labels_cpu = labels.numpy()

            # Save array as Pickle
            results = {f"embedding": embedding_cpu, "labels": labels_cpu}
            utils.save_embedding(results, self.config)
            self.logger.log(msg=f"Embedding Size: {embedding.shape}.")

            # Save Model
            utils.save_model(self.model, id=self.config.meta.id)
            self.logger.log(msg=f"Model Saved!")

            # Stop Tracking Emissions


def main(cfg):

    exp = Experiment(cfg, logger=mylogger, dev=False)
    # Logging
    exp.logger.log(msg=f"Starting Experiment {exp.config.meta.id}")
    exp.logger.log(
        msg=f"{exp.config.model_params.module} training on {exp.config.data_params.module}"
    )
    exp.logger.log(f"{utils.count_parameters(exp.model)} trainable parameters")

    # Execute Experiment
    exp.run()
    exp.save_run()
    exp.logger.log("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="Identifier for config `yaml`.",
    )
    args = parser.parse_args()
    this = sys.modules[__name__]

    main(args.config_file)
