import torch
from torch import nn


class LinearEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int = 48, out_channels: int = 96):
        """
        For a SWIN transformer, we need 96 channels as opposed to whichever channels we had before
        """
        super().__init__()
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        return self.linear(x)
