import torch
from torch import nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D

from torchvision.transforms import ToPILImage


class PositionalEmbedding(nn.Module):
    def __init__(self, m, n):
        super().__init__()

    def mesh(self, x):
        # Suppose x (56, 56, 96)

        # Need a 3d Mesh of size (56, 56, 96)
        mesh = torch.arange(x.shape[0] * x.shape[1] * x.shape[2])
        mesh = mesh.reshape(x.shape)

        return mesh

    def encode(self, x):
        cos = torch.cos(x)
        sin = torch.sin(x)

        return sin + cos

    @staticmethod
    def pre_made(x):
        """
        Uses the premade sinusoid encoding for 2D transformers
        """

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        cls = PositionalEncoding2D(x.shape[-1])  # init channels
        return cls(x)


if __name__ == "__main__":
    x = torch.randn((56, 56, 96))

    pos_embed = PositionalEmbedding(1, 1)

    mesh = pos_embed.mesh(x)
    encoded = pos_embed.pre_made(x)
    breakpoint()
    encoded = pos_embed.encode(mesh)
