
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: int, 
                stride: int=1, 
                padding: int=0, 
                use_activation: bool=True, 
                use_norm: bool=True, 
                upsample: bool=False
                ) -> None:

        super(ConvBlock, self).__init__()

        if not upsample:
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="reflect")
        else:
            conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1)

        self.layers = nn.Sequential(
            conv_layer,
            nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True) if use_activation else nn.Identity()
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int=1, padding: int=0, use_norm: bool=True) -> None:
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, kernel_size=3, padding=1, use_activation=False)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return F.relu(self.layers(x) + x)


class Generator(nn.Module):
    def __init__(self, num_res_blocks: int=3) -> None:
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=1, padding=3, use_norm=False),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256, kernel_size=3, stride=2, padding=1) for _ in range(num_res_blocks)]
        )

        self.decoder = nn.Sequential(
            ConvBlock(256, 128, kernel_size=3, stride=2, padding=1, upsample=True),
            ConvBlock(128, 64, kernel_size=3, stride=2, padding=1, upsample=True),
            ConvBlock(64, 3, kernel_size=7, stride=1, padding=3, use_activation=False, use_norm=False)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)

        return torch.tanh(x)



"""
dimension testing:
x = torch.rand((1, 3, 128, 128))
model = Generator(num_res_blocks=6)

out = model(x)
"""

