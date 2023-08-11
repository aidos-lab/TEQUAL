import torch.nn.functional as F
from torch import nn

from models.base_model import BaseModel
from loaders.factory import register
from dataclasses import dataclass


@dataclass
class NNConfig:
    module: str = "models.custom_model"
    input_size: int = 28 * 28
    hidden1_size: int = 250
    hidden2_size: int = 250
    output_size: int = 10


class NN(BaseModel):
    def __init__(self, config: NNConfig):
        super(NN, self).__init__(config)
        self.linear1 = nn.Linear(config.input_size, config.hidden1_size)
        self.linear2 = nn.Linear(config.hidden1_size, config.hidden2_size)
        self.linear3 = nn.Linear(config.hidden2_size, config.output_size)

    def forward(self, X):
        X = X.view(-1, 28 * 28)
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)


def initialize():
    register("model", NN)
