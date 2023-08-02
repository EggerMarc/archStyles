import torch
import torchvision


"""
TODO:
1. Generalize mirroring of Encoder OR Generalize target shape
"""


class Decoder(torch.nn.Module):
    def __init__(self, n_features, target_size=None):
        super().__init__()

        self.decoder = torch.nn.Sequential()

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    from image import train_loader

    train_loader = train_loader()
    model = Decoder(100)
    breakpoint()
