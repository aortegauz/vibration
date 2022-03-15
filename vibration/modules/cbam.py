import torch
from torch import Tensor
from torch import nn

class ChannelAttention(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output


class SpatialAttention(nn.Module):
    
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca=ChannelAttention(in_channels, reduction)
        self.sa=SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+x