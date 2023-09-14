import torch
import torchvision

"""
TODO:
1. Calculate LazyLinear in_features +
2. Build optimal architecture. Beware of breaking zsh due to init in large params
"""


class Encoder(torch.nn.Module):
    def __init__(self, K, n_features):
        super().__init__()

        self.encoder = torch.nn.Sequential()

        in_channels = 3
        out_channels = 4
        for k in range(K):
            out_channels = out_channels * 2
            self.encoder.add_module(
                f"Convolutional Layer {k}_1",
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    padding=1,
                    stride=2,
                ),
            )
            in_channels = out_channels
            self.encoder.add_module(f"ReLU Layer {k}_1", torch.nn.ReLU())
            self.encoder.add_module(
                f"Convolutional Layer {k}_2",
                torch.nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
            self.encoder.add_module(f"ReLU Layer {k}_2", torch.nn.ReLU())

        ## LazyLinear in_features
        input_in_features = torch.randn(1, 3, 256, 256)
        output_in_features = self.encoder(input_in_features)
        in_features = output_in_features.numel() // output_in_features.shape[0]

        self.encoder.add_module(
            f"Flatten", torch.nn.Flatten()
        )
        self.encoder.add_module(
            f"Linear Layer 1",
            torch.nn.Linear(in_features, out_features=n_features),
        )
        self.encoder.add_module(f"ReLU Layer {K}", torch.nn.ReLU())
        self.encoder.add_module(
            f"Linear Layer 2",
            torch.nn.Linear(in_features=n_features, out_features=n_features),
        )

        self.encoder.add_module(f"Softmax Layer", torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    from image import train_loader

    train_loader = train_loader()
    model = Encoder(5, 100)
    breakpoint()
    v = model(iter(train_loader))
    breakpoint()
