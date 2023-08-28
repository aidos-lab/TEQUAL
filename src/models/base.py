from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor, nn

from config import AutoEncoderConfig

from .types_ import *


class BaseVAE(nn.Module):
    def __init__(self, config: AutoEncoderConfig):
        super(BaseVAE, self).__init__()
        self.config = config

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvT_BN_LRelu, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        )
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.LeakyReLU())

    def forward(self, x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return x
