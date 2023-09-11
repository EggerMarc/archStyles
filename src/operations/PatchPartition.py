import torch
from torch import nn
from einops import rearrange


class PatchPartition(torch.nn.Module):
    def __init__(self, _height: int = 4, _width: int = 4):
        """
        The whole image is divided into regular non-overlapping patches of 4×4 size

        Should take in an image
        We can take the partitions in three ways:
            1. flatten and them and then take [n+m:n+m+4] for m in range(img.height), with n being patch id
            2. for m in range(img.height) take [n:n+4] -> Let's opt for this first, as it maintains shape
            3. use einops for rearranging data:
                einops.rearrange(x.feature_maps[0], "B C H W -> B H/4 W/4 C*16")
            4. simply use a conv2d, with out_channels =


        Return something of size
            img.height / 4, img.width / 4, 16 * 3. 16 * 3 for channels because we have 16 pixels, each with 3 channels
        """
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels=3,
            out_channels=_height
            * _width
            * 3,  # We can already do the Linear Embedding here
            kernel_size=(_height, _width),  # 4x4 selection
            stride=(_height, _width),  # move 4x4
        )
        self.shape = (_height, _width, _height * _width * 3)

    def forward(self, x):
        return self.convolution(x).transpose(0, -1)
