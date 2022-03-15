from torch import nn, Tensor, flatten
from typing import List, Tuple
from .modules.cbam import CBAMBlock
from math import prod

class ResidualBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        # Residual Path
        self.channel_matching = nn.Identity()
        if in_channels != out_channels:
            self.channel_matching =  nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Convolutional layers
        padding = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.channel_matching(x)
        out += residual
        out = self.relu(out)
        return out


class MainBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        pooling: List[int] = [2,2],
        dropout: int = 0,
        cbam: bool = False,
    ) -> None:
        super().__init__()
        self.resconv = ResidualBlock(in_channels, out_channels, kernel_size)
        self.cbam = CBAMBlock(out_channels) if cbam else nn.Identity()
        self.avgpooling = nn.AvgPool2d(pooling)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.resconv(x)
        out = self.cbam(out)
        out = self.dropout(out)
        out = self.avgpooling(out)
        return out


class MainBlockTranspose(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        scale: Tuple[int] = (2,2),
        dropout: int = 0,
        cbam: bool = False,
    ) -> None:
        super().__init__()
        self.resconv = ResidualBlock(in_channels, out_channels, kernel_size)
        self.cbam = CBAMBlock(out_channels) if cbam else nn.Identity()
        self.upsampling = nn.Upsample(scale_factor=scale, mode='nearest')
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsampling(x)
        out = self.resconv(out)
        out = self.cbam(out)
        out = self.dropout(out)
        return out


class Net(nn.Module):

    def __init__(
        self,
        input_size: List[int],
        in_channels: int,
        code_dim: int,
        channels: List[int],
        kernel_size: List[int],
        freq_pooling: List[int],
        time_pooling: List[int],
        dropout: int = 0,
        cbam: bool = False,
    ) -> None:
        super(Net,self).__init__()
        total_channels = [in_channels] + channels
        n_layers = len(channels)
        # Encoder
        encoder = []
        for i in range(n_layers):
            encoder.append(
                MainBlock( 
                    total_channels[i],
                    total_channels[i+1],
                    kernel_size[i],
                    [freq_pooling[i], time_pooling[i]],
                    dropout,
                    cbam,
                )
            )
        self.encoder = nn.Sequential(*encoder)

        # Code
        size_code = input_size[0]*input_size[1]*channels[-1]\
            //(prod(freq_pooling)*prod(time_pooling))
        self.code = nn.Sequential(
            nn.Linear(size_code, code_dim),
            nn.BatchNorm1d(code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, size_code),
            nn.BatchNorm1d(size_code),
            nn.ReLU(),
        )

        # Decoder
        decoder = []
        for i in range(n_layers-1,0,-1):
            decoder.append(
                MainBlockTranspose(
                    total_channels[i+1],
                    total_channels[i],
                    kernel_size[i],
                    (freq_pooling[i], time_pooling[i]),
                    dropout,
                    cbam,
                )
            )
        decoder.append(nn.Upsample(scale_factor=(freq_pooling[0], time_pooling[0]), mode='nearest'))
        decoder.append(nn.Conv2d(channels[0], in_channels, kernel_size[0], padding=(kernel_size[0]-1)//2))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        shape = z.shape
        z = flatten(z, start_dim=1, end_dim=- 1)
        z = self.code(z)
        z = z.reshape(shape)
        y = self.decoder(z)
        return y