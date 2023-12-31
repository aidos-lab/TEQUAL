import torch
from torch import nn
from torch.nn import functional as F

from config import AutoEncoderConfig
from loaders.factory import register
from models import BaseVAE

from .types_ import *


class LinearVAE(BaseVAE):
    def __init__(self, config) -> None:
        super(LinearVAE, self).__init__(config)
        self.latent_dim = self.config.latent_dim
        self.hidden_dims = self.config.hidden_dims
        self.in_channels = self.config.in_channels
        self.img_size = self.config.img_size
        self.input_dim = self.img_size**2

        modules = []

        # Build Encoder Architechture
        in_features = self.input_dim
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                )
            )
            in_features = h_dim
        self.encoder = nn.Sequential(*modules)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        # VAE Layers
        self.fc_mu = nn.Linear(in_features, self.latent_dim)
        self.fc_var = nn.Linear(in_features, self.latent_dim)

        # Build Decoder
        self.hidden_dims.reverse()
        modules = []

        in_features = self.latent_dim
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                )
            )
            in_features = h_dim
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(in_features, self.input_dim),
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        x = torch.flatten(input, start_dim=1)
        z = self.encoder(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        x = self.decoder(z)
        result = self.final_layer(x)
        result.view((-1, 1, self.img_size, self.img_size))
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

        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        input = torch.flatten(input, start_dim=1)

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

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

    def latent(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z


def initialize():
    register("model", LinearVAE)
