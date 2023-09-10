import torch
import torchvision


"""
TODO:
1. Generalize mirroring of Encoder OR Generalize target shape
"""


class Decoder(torch.nn.Module):
    def __init__(self, K, n_features, target_size=None):
        super().__init__()

        self.decoder = torch.nn.Sequential()

        # Here we want to mirror the encoder

        self.decoder.add_module(
            f"Linear 1", torch.nn.Linear(in_features=n_features, out_features=6272)
        )

        self.decoder.add_module(f"ReLU Activation 0", torch.nn.ReLU())
        self.decoder.add_module(
            f"Linear 2", torch.nn.Linear(in_features=6272, out_features=6272)
        )
        self.decoder.add_module(f"Unflatten", torch.nn.Unflatten(1, (128, 7, 7)))

        """
        self.decoder.add_module(
            f"Conv Trial",
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=5, stride=2, padding=1, output_padding=1
            ),
        )
        """
        out_channels = 4 * 2 ** (K)
        for k in range(K):
            in_channels = out_channels
            out_channels = int(out_channels / 2)
            self.decoder.add_module(
                f"Convolutional Transpose Layer {k}_1",
                torch.nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels if k != K - 1 else 3,
                    kernel_size=5,
                    padding=1,
                    stride=2,
                    output_padding=1,
                ),
            )

            self.decoder.add_module(f"ReLU Layer {k}_1", torch.nn.ReLU())

            self.decoder.add_module(
                f"Convolutional Layer {k}_2",
                torch.nn.ConvTranspose2d(
                    in_channels=out_channels if k != K - 1 else 3,
                    out_channels=out_channels if k != K - 1 else 3,
                    kernel_size=3,
                    padding=3,
                    stride=1,  # Stride should be kept at 1
                    dilation=2,
                    output_padding=1,
                ),
            )
            self.decoder.add_module(f"ReLU Layer {k}_2", torch.nn.ReLU())

        self.decoder.add_module(
            f"ConvTranspose {K}",
            torch.nn.ConvTranspose2d(
                in_channels=3,
                out_channels=3,
                kernel_size=2,
                padding=0,
                stride=1,
                dilation=1,
                output_padding=0,
            ),
        )

        self.decoder.add_module(f"Activation {K}", torch.nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    from image import train_loader
    from encoder import Encoder

    train_loader = train_loader()
    encoder = Encoder(5, 100)
    encoded = encoder(next(iter(train_loader)))
    model = Decoder(5, 100)

    v = model(encoded)
    breakpoint()
