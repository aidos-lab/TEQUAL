import argparse
import sys
import time

import torch
from omegaconf import OmegaConf

import loaders.factory as loader
import utils
from loggers.logger import Logger, timing
from metrics.metrics import compute_acc

torch._C._mps_emptyCache()

my_experiment = utils.read_parameter_file().experiment
mylogger = Logger(exp=my_experiment, name="training_results")


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
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.logger.wandb_init(self.config.meta)

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

    @timing(mylogger)
    def run(self):
        """
        Runs an experiment given the loaded config files.
        """
        start = time.time()
        for epoch in range(self.config.trainer_params.num_epochs):
            stats = self.run_epoch()
            if epoch % 10 == 0:
                end = time.time()
                self.logger.log(
                    msg=f"Training the model 10 epochs took: {end - start:.2f} seconds."
                )
                reported_loss = self.loss.item()
                self.logger.log(msg=f"Loss: {reported_loss}")
                if "Reconstruction_Loss" in stats.keys():
                    recon_loss = stats["Reconstruction_Loss"]
                    self.logger.log(msg=f"Reconstruction Loss: {recon_loss.item()}")

                start = time.time()

        # self.finalize_run()

    def run_epoch(self):
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

            return stats

        # Delete Train Loader
        del loader

    @timing(mylogger)
    def finalize_run(self):
        # Train Reconstruction Loss
        loss, acc = compute_acc(self.model, self.dm.test_dataloader(), self.loss_fn)

        # Test Reconstruction Loss

        # Val Reconstruction Loss

        # Log statements
        self.logger.log(
            f"Test accuracy {acc:.2f},\n Confusion Matrix:\n {cfm}.",
            params={
                "test_acc": acc,
                "test_loss": loss,
            },
        )

    def compute_metrics(self, epoch):

        loss, acc = compute_acc(self.model, self.dm.val_dataloader(), self.loss_fn)

        # Log statements to console
        self.logger.log(
            msg=f"epoch {epoch} | train loss {loss.item():.2f} | Accuracy {acc:.2f}",
            params={"epoch": epoch, "val_loss": loss.item(), "val_acc": acc},
        )

    @timing(mylogger)
    def save_run(self):

        embedding = torch.Tensor()
        labels = torch.Tensor()
        if self.loss:
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

        else:
            self.logger.log(msg="Not saving embedding: Nans in loss function")


def main(cfg):

    exp = Experiment(cfg, logger=mylogger, dev=True)
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
