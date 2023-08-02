import torch
import torchvision

"""
TODO:
1. Build optimal architecture. Beware of breaking zsh due to init in large params
"""


class Encoder(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.encoder = torch.nn.Sequential()

        # for k in range(K):
        #    self.encoder.add_module(f"Convolutional Layer {k}")

    def forward(self, x):
        print(x.size())
        return self.encoder(x)


if __name__ == "__main__":
    from image import train_loader

    train_loader = train_loader()
    model = Encoder(100)
    breakpoint()
