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
                    kernel_size=3,
                    stride=2,
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
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
        self.decoder = nn.Sequential(*modules)

        # FINAL LAYER
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                self.rhidden_dims[-1],
                self.rhidden_dims[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.rhidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.rhidden_dims[-1],
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
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


def stretch(X, alpha, gamma, beta, moving_mag, moving_min, eps, momentum, training):
    """
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    """
    if not training:
        X_hat = (X - moving_min) / moving_mag
    else:
        assert len(X.shape) in (2, 4)
        min_ = X.min(dim=0)[0]
        max_ = X.max(dim=0)[0]

        mag_ = max_ - min_
        X_hat = (X - min_) / mag_
        moving_mag = momentum * moving_mag + (1.0 - momentum) * mag_
        moving_min = momentum * moving_min + (1.0 - momentum) * min_
    Y = (X_hat * gamma * alpha) + beta
    return Y, moving_mag.data, moving_min.data


class Stretch(nn.Module):
    """
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    """

    def __init__(self, num_features, num_dims, alpha):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.alpha = alpha
        self.gamma = nn.Parameter(0.01 * torch.ones(shape))
        self.beta = nn.Parameter(np.pi * torch.ones(shape))
        self.register_buffer("moving_mag", 1.0 * torch.ones(shape))
        self.register_buffer("moving_min", np.pi * torch.ones(shape))

    def forward(self, X):
        if self.moving_mag.device != X.device:
            self.moving_mag = self.moving_mag.to(X.device)
            self.moving_min = self.moving_min.to(X.device)
        Y, self.moving_mag, self.moving_min = stretch(
            X,
            self.alpha,
            self.gamma,
            self.beta,
            self.moving_mag,
            self.moving_min,
            eps=1e-5,
            momentum=0.99,
            training=self.training,
        )
        return Y


class Conv_BN_LRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        super(Conv_BN_LRelu, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        )
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.LeakyReLU())

    def forward(self, x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return x


class ConvT_BN_LRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding=0,
    ):
        super(ConvT_BN_LRelu, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        )
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.LeakyReLU())

    def forward(self, x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return x


def initialize():
    register("model", DGAECONV)
