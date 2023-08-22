import torch
from omegaconf import OmegaConf
import os

from generate_experiments import generate_experiments

import utils

from loggers.logger import Logger, timing
from metrics.metrics import compute_confusion, compute_acc

import loaders.factory as loader
import time

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

        self.logger.log("Setup")
        self.logger.wandb_init(self.config.meta)

        # Load the dataset
        self.dm = loader.load_module("dataset", self.config.data)

        # Load the model
        model = loader.load_module("model", self.config.model)

        # Send model to device
        self.model = model.to(self.device)

        # Loss function and optimizer.
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.trainer.lr)

        # Log info
        self.logger.log(
            f"{self.config.model.module} has {utils.count_parameters(self.model)} trainable parameters"
        )

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
                self.compute_metrics(epoch)
                self.logger.log(
                    msg=f"Training the model 10 epochs took: {end - start:.2f} seconds."
                )
                start = time.time()

        self.finalize_run()

    def run_epoch(self):
        for batch_idx, (x, y) in enumerate(self.dm.train_dataloader()):
            batch_gpu, _ = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            results = self.model(batch_gpu, labels=y)
            pred = results[0]
            print("First Prediction:")

            loss = self.loss_fn(pred, batch_gpu)
            # loss = self.model.loss_function(
            #     *results,
            #     batch_gpu,
            #     batch_idx=batch_idx,
            #     M_N=0.00025,
            # )
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

    def save_run(self):
        embedding = []
        for sample, _ in self.dm.entire_ds:
            x = self.model.encode(sample)
            embedding.append(x)

        # Save array as Pickle
        utils.save_embedding(embedding, self.config)


def main():
    path = utils.get_experiment_path()
    experiments = os.listdir(path)
    for cfg in experiments:
        file = os.path.join(path, cfg)
        print(f"Starting experiments for :{file}")
        exp = Experiment(file, logger=mylogger, dev=True)
        exp.run()
        exp.save_run()


if __name__ == "__main__":
    main()
