import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import AutoEncoderConfig
from loaders.factory import register
from models import *

from .types_ import *


class DGAECONV(BaseVAE):
    def __init__(self, config: AutoEncoderConfig) -> None:
        super(DGAECONV, self).__init__(config)
        self.latent_dim = self.config.latent_dim
        self.hidden_dims = self.config.hidden_dims
        self.in_channels = self.config.in_channels
        self.img_size = self.config.img_size
        self.input_dim = (1, self.img_size, self.img_size)

        # Fixed for now
        self.fc_hidden = 64
        self.alpha = 3
        self.kernel_size = 10

        _, m, n = self.input_dim

        self.encoder_seq = nn.ModuleList()

        for idx in range(len(self.hidden_dims) - 1):
            self.encoder_seq.append(
                Conv_BN_LRelu(
                    self.hidden_dims[idx],
                    self.hidden_dims[idx + 1],
                    self.kernel_size,
                    4,
                    padding=1,
                )
            )

        self.en_fc = nn.Linear(self.hidden_dims[-1] * 4 * 4, self.fc_hidden)
        self.to_lat = nn.Linear(self.fc_hidden, self.latent_dim)
        self.strecth = Stretch(self.latent_dim, 2, self.alpha)

        self.to_dec = nn.Linear(self.latent_dim * 2, self.fc_hidden)
        self.de_fc = nn.Linear(self.fc_hidden, self.hidden_dims[-1] * 4 * 4)

        self.rhidden_dims = self.hidden_dims[::-1]

        self.decoder_seq = nn.ModuleList()

        for idx in range(len(self.rhidden_dims) - 1):
            self.decoder_seq.append(
                ConvT_BN_LRelu(
                    self.rhidden_dims[idx],
                    self.rhidden_dims[idx + 1],
                    self.kernel_size,
                    4,
                    padding=1,
                )
            )

        self.decoder_seq.append(
            nn.Conv2d(self.rhidden_dims[-1], self.rhidden_dims[-1], 3, padding=1)
        )
        self.decoder_seq.append(nn.Sigmoid())

    def sample(self, num_samples=100, z=None):
        c = torch.cat((torch.cos(2 * np.pi * z), torch.sin(2 * np.pi * z)), 0)
        c = c.T.reshape(self.latent_dim * 2, -1).T
        samples = self.decode(c)
        return samples

    def reconstr(self, x):
        z = self.encode(x)
        c = torch.cat((torch.cos(2 * np.pi * z), torch.sin(2 * np.pi * z)), 0)
        c = c.T.reshape(self.latent_dim * 2, -1).T
        reconstr = self.decode(c)
        return reconstr

    def encode(self, x):
        for idx in range(len(self.encoder_seq)):
            x = self.encoder_seq[idx](x)

        x = torch.flatten(x, start_dim=1)

        x = self.en_fc(x)

        z = self.to_lat(x)
        s = self.strecth(z)

        return s

    def latent(self, x):
        z = self.encode(x)
        return z

    def decode(self, x):
        x = nn.LeakyReLU()(self.to_dec(x))
        x = nn.LeakyReLU()(self.de_fc(x))
        x = x.view(-1, self.hidden_dims[-1], 4, 4)

        for idx in range(len(self.decoder_seq)):
            x = self.decoder_seq[idx](x)
        return x

    def reparameterize(self, z, **args):
        diff = torch.abs(z - z.unsqueeze(axis=1))
        none_zeros = torch.where(diff == 0.0, torch.tensor([100.0]).to(z.device), diff)
        z_scores, _ = torch.min(none_zeros, axis=1)
        std = torch.normal(mean=0.0, std=1.0 * z_scores).to(z.device)
        s = z + std
        c = torch.cat((torch.cos(2 * np.pi * s), torch.sin(2 * np.pi * s)), 0)
        c = c.T.reshape(self.latent_dim * 2, -1).T
        return c

    def forward(self, x):
        z = self.encode(x)
        c = self.reparameterize(z)
        reconstr = self.decode(c)
        return [reconstr, x, c, z]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    def loss_function(self, *args, **kwargs) -> dict:
        reconstr = args[0]
        x = args[1]
        _ = args[2]
        _ = args[3]

        fn = nn.BCELoss()

        loss = fn(reconstr, x)

        return {
            "loss": loss,
        }


def initialize():
    register("model", DGAECONV)
