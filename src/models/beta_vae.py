import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F

from loaders.factory import register
from models import BaseVAE

from .types_ import *


class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        config,
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "H",
        **kwargs,
    ) -> None:
        super(BetaVAE, self).__init__(config)
        self.latent_dim = self.config.latent_dim
        self.hidden_dims = self.config.hidden_dims
        self.in_channels = self.config.in_channels
        self.img_size = self.config.img_size
        self.input_dim = self.img_size**2

        # Specific Params
        self.beta = self.config.alpha
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []

        # Build Encoder Architechture
        for idx in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.hidden_dims[idx],
                        self.hidden_dims[idx + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.hidden_dims[idx + 1]),
                    nn.LeakyReLU(),
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
        self.fc_mu = nn.Linear(self.num_features, self.latent_dim)
        self.fc_var = nn.Linear(self.num_features, self.latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, self.num_features)

        self.rhidden_dims = self.hidden_dims[::-1]

        for i in range(len(self.rhidden_dims) - 1):
            layer = nn.Sequential(
                nn.ConvTranspose2d(
                    self.rhidden_dims[i],
                    self.rhidden_dims[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(self.rhidden_dims[i + 1]),
                nn.LeakyReLU(),
            )
            modules.append(layer)

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

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.encoded_shape)
        result = self.decoder(result)

        # TODO: Figure out how/why this final layer works
        # result = self.final_layer(result)
        # result = result.view(-1, self.img_size, self.img_size)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, z, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": kld_loss,
        }

    def latent(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def initialize():
    register("model", BetaVAE)
