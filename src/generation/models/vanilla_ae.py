import torch.nn.functional as F
from torch import nn

from models import BaseVAE
from loaders.factory import register


class VanillaAE(BaseVAE):
    def __init__(self, config):
        super(VanillaAE, self).__init__(config)

        self.encoder = nn.Sequential(
            nn.Linear(config.input_size, config.hidden1_size),
            nn.ReLU(),
            nn.Linear(config.hidden1_size, config.hidden2_size),
            nn.ReLU(),
            nn.Linear(config.hidden2_size, config.hidden3_size),
            nn.ReLU(),
            nn.Linear(config.hidden3_size, config.hidden4_size),
            nn.ReLU(),
            nn.Linear(config.hidden4_size, config.hidden5_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden5_size, config.hidden4_size),
            nn.ReLU(),
            nn.Linear(config.hidden4_size, config.hidden3_size),
            nn.ReLU(),
            nn.Linear(config.hidden3_size, config.hidden2_size),
            nn.ReLU(),
            nn.Linear(config.hidden2_size, config.hidden1_size),
            nn.ReLU(),
            nn.Linear(config.hidden1_size, config.input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def initialize():
    register("model", VanillaAE)
