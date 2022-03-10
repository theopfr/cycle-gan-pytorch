
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import device


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, use_activation: bool=True, use_norm: bool=True) -> None:
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.conv_block(x)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, noise_rate: float=0.0) -> None:
        super(GaussianNoiseLayer, self).__init__()

        self.noise_rate = noise_rate

    def forward(self, x):
        return x + (self.noise_rate * torch.rand_like(x, requires_grad=False).to(device))


class Discriminator(nn.Module):
    def __init__(self, gaussian_noise_rate: float=0.0) -> None:
        super(Discriminator, self).__init__()

        self.gaussianNoiseLayer = GaussianNoiseLayer(noise_rate=gaussian_noise_rate)

        self.conv1 = ConvBlock(3, 64, kernel_size=4, stride=2, padding=1, use_norm=False)
        self.conv2 = ConvBlock(64, 128, kernel_size=4, stride=2)
        self.conv3 = ConvBlock(128, 256, kernel_size=4, stride=2)
        self.conv4 = ConvBlock(256, 512, kernel_size=4, stride=2)
        self.conv5 = ConvBlock(512, 1, kernel_size=4, stride=1, padding=1, use_activation=False, use_norm=False)

        self.maxpool = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(512 * 3 * 3, 128)
        self.linear2 = nn.Linear(128, 1)


    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.gaussianNoiseLayer(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        #x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        #x = F.relu(self.linear1(x))

        return torch.sigmoid(x)


"""
dimension testing:
x = torch.rand((1, 3, 128, 128))
model = Discriminator()

out = model(x)
print(out.shape)
"""
