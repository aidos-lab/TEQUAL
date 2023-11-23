import functools
import operator

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
        self.kernel_size = self.config.kernel_size

        # Fixed for now
        self.fc_hidden = 64
        self.alpha = self.config.alpha

        _, m, n = self.input_dim

        modules = []

        for idx in range(len(self.hidden_dims) - 1):
            modules.append(
                Conv_BN_LRelu(
                    self.hidden_dims[idx],
                    self.hidden_dims[idx + 1],
                    self.kernel_size,
                    4,
                    padding=1,
                )
            )

        self.encoder = nn.Sequential(*modules)
        # Tracking Encoder Shapes
        self.encoded_shape = self.encoder(
            torch.rand(1, self.in_channels, self.img_size, self.img_size)
        ).shape[1:]
        self.num_features = functools.reduce(
            operator.mul,
            list(self.encoded_shape),
        )

        # VAE Linear Layers
        self.en_fc = nn.Linear(self.num_features, self.fc_hidden)
        self.to_lat = nn.Linear(self.fc_hidden, self.latent_dim)

        # Batch Norm from:
        # http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
        self.strecth = Stretch(self.latent_dim, 2, self.alpha)

        # VAE Linear Layers
        self.to_dec = nn.Linear(self.latent_dim * 2, self.fc_hidden)
        self.de_fc = nn.Linear(self.fc_hidden, self.num_features)

        # DECODER
        self.rhidden_dims = self.hidden_dims[::-1]

        modules = []

        for idx in range(len(self.rhidden_dims) - 1):
            modules.append(
                ConvT_BN_LRelu(
                    self.rhidden_dims[idx],
                    self.rhidden_dims[idx + 1],
                    self.kernel_size,
                    4,
                    padding=1,
                )
            )
        self.decoder = nn.Sequential(*modules)

        # FINAL LAYER
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                self.rhidden_dims[-1],
                self.rhidden_dims[-1],
                kernel_size=3,
                stride=4,
                padding=1,
            ),
            nn.BatchNorm2d(self.rhidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.rhidden_dims[-1],
                out_channels=self.in_channels,
                kernel_size=3,
                stride=4,
                padding=8,
            ),
            nn.Tanh(),
        )

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
        x = self.encoder(x)

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
        x = x.view(-1, *self.encoded_shape)

        x = self.decoder(x)
        x = self.final_layer(x)
        # x = x.view(-1, self.img_size, self.img_size)
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
