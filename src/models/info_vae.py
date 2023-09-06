import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F

from loaders.factory import register
from models import BaseVAE

from .types_ import *


class InfoVAE(BaseVAE):
    def __init__(
        self,
        config,
        alpha: float = -0.5,
        beta: float = 5.0,
        reg_weight: int = 100,
        kernel_type: str = "imq",
        latent_var: float = 2.0,
        **kwargs
    ) -> None:
        super(InfoVAE, self).__init__(config)
        self.latent_dim = self.config.latent_dim
        self.hidden_dims = self.config.hidden_dims
        self.in_channels = self.config.in_channels
        self.img_size = self.config.img_size
        self.input_dim = self.img_size**2

        # Specific Params
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        assert alpha <= 0, "alpha must be negative or zero."

        self.alpha = alpha
        self.beta = beta

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
            torch.rand(1, 1, self.img_size, self.img_size)
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
        recons = args[0]
        input = args[1]
        z = args[2]
        mu = args[3]
        log_var = args[4]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = (
            self.beta * recons_loss
            + (1.0 - self.alpha) * kld_weight * kld_loss
            + (self.alpha + self.reg_weight - 1.0) / bias_corr * mmd_loss
        )
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "MMD": mmd_loss,
            "KLD": -kld_loss,
        }

    def compute_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == "rbf":
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == "imq":
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError("Undefined kernel type.")

        return result

    def compute_rbf(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(
        self, x1: Tensor, x2: Tensor, eps: float = 1e-7
    ) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean()
        return mmd

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

    def latent(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def initialize():
    register("model", InfoVAE)
