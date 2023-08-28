import os
import time

import torch
from omegaconf import OmegaConf

import loaders.factory as loader
import utils
from loggers.logger import Logger, timing
from metrics.metrics import compute_acc, compute_confusion

torch.cuda.empty_cache()

mylogger = Logger()


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.logger.log("Setup")
        self.logger.wandb_init(self.config.meta)

        # Load the dataset
        self.dm = loader.load_module("dataset", self.config.data)

        # Set model input size
        self.config.model.img_size = self.config.data.img_size

        # Load the model
        model = loader.load_module("model", self.config.model)

        # Send model to device
        self.model = model.to(self.device)

        # Loss function and optimizer.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.trainer.lr)

    @timing(mylogger)
    def run(self):
        """
        Runs an experiment given the loaded config files.
        """
        start = time.time()
        for epoch in range(self.config.trainer.num_epochs):
            self.run_epoch()

            if epoch % 10 == 0:
                end = time.time()
                # self.compute_metrics(epoch)
                self.logger.log(
                    msg=f"Training the model 10 epochs took: {end - start:.2f} seconds."
                )
                start = time.time()

        # self.finalize_run()

    def run_epoch(self):
        for batch_idx, (x, y) in enumerate(self.dm.train_dataloader()):
            X, _ = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            results = self.model(X)

            stats = self.model.loss_function(
                *results,
                batch_idx=batch_idx,
                M_N=0.00025,
            )
            loss = stats["loss"]
            loss.backward()

            self.optimizer.step()

    @timing(mylogger)
    def finalize_run(self):
        # Compute accuracy
        loss, acc = compute_acc(self.model, self.dm.test_dataloader(), self.loss_fn)

        # Compute confusion
        cfm = compute_confusion(self.model, self.dm.test_dataloader())

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

        # Save training or full?
        for train_data, train_labels in self.dm.full_dataloader():

            embedding = self.model.latent(train_data).detach().numpy()
            labels = train_labels.detach().numpy()

            results = {"embedding": embedding, "labels": labels}

        # Save array as Pickle
        utils.save_embedding(results, self.config)
        self.logger.log(msg=f"Embedding Size: {embedding.shape}.")


def main():
    path = utils.get_experiment_dir()
    experiments = os.listdir(path)
    experiments.sort()
    for cfg in experiments:
        file = os.path.join(path, cfg)
        exp = Experiment(file, logger=mylogger, dev=True)
        # Logging
        exp.logger.log(msg=f"Starting Experiments for {cfg}")
        exp.logger.log(
            msg=f"{exp.config.model.module} training on {exp.config.data.module}"
        )
        exp.logger.log(f"{utils.count_parameters(exp.model)} trainable parameters")

        # Execute Experiment
        exp.run()
        exp.save_run()
        print()


if __name__ == "__main__":
    main()
