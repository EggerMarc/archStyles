import torch
import torchvision
from image import ArchImages
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = ArchImages(
    "../images",
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(1000, antialias=True),
            torchvision.transforms.CenterCrop(1000),
        ]
    ),
    device=device,
)


class Model_0_2(torch.nn.Module):
    def __init__(self, input_channels, K, N):
        super(Model_0_2, self).__init__()

        encoder_layers = []
        decoder_layers = []
        input_size = input_channels

        # Encoder
        for i in range(K):
            encoder_layers.append(
                torch.nn.Conv2d(input_size, 2 ** (i + 6), kernel_size=3, padding=1)
            )
            encoder_layers.append(torch.nn.ReLU())
            encoder_layers.append(torch.nn.MaxPool2d(2, 2, padding=1))
            input_size = 2 ** (i + 6)

        encoder_layers.append(torch.nn.Conv2d(input_size, N, kernel_size=3, padding=1))

        # Decoder
        for i in range(K - 1, -1, -1):
            decoder_layers.append(
                torch.nn.Conv2d(N, 2 ** (i + 6), kernel_size=3, padding=1)
            )
            decoder_layers.append(torch.nn.ReLU())
            decoder_layers.append(torch.nn.Upsample(scale_factor=2))

        decoder_layers.append(
            torch.nn.Conv2d(2 ** (K + 5), input_channels, kernel_size=3, padding=1)
        )
        decoder_layers.append(torch.nn.Sigmoid())

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Model_0_1(torch.nn.Module):
    def __init__(self, K, n_features):
        """
        K int: num convolutions
        n_features: hidden features
        """
        super().__init__()
        self.K = K
        self.n_features = n_features

    def encoder(self):
        encode = torch.nn.Sequential()
        encode.add_module(
            f"Convolution 0",
            torch.nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...),
        )
        encode.add_module(f"Activation 0", torch.nn.Tanh())

        for k in self.K:
            encode.add_module(
                f"Convolution {k + 1}",
                torch.nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...),
            )
            encode.add_module(f"Activation {k + 1}", torch.nn.Tanh())

        encode.add_module(
            f"Feature Vector",
            torch.nn.LazyLinear(out_features=self.n_features),
        )

        encode.add_module(f"SoftMax Layer", torch.nn.Softmax())

        return encode

    def decoder(self):
        decode = torch.nn.Sequential()
        decode.add_module(
            f"Decoding Convolution 0",
            torch.nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...),
        )
        decode.add_module("Activation 0", torch.nn.Tanh())

        for k in self.K:
            decode.add_module(
                f"Convolution {k + 1}",
                torch.nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...),
            )
            decode.add_module(f"Activation {k + 1}", torch.nn.Tanh())

    def forward(self, X):
        features = self.encoder(X)
        out = self.decoder(features)

        return out


if __name__ == "__main__":
    train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()

    # model = Model_0_1(K=10, n_features=100)
    model = Model_0_2(input_channels=3, K=5, N=100).to(device=device)

    optim = torch.optim.Adam(params=model.parameters(), lr=1e-100)

    epochs = 10
    epoch_loss = []
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, data in enumerate(train_loader):
            inputs = data.to(device)
            # Zero the gradients
            optim.zero_grad()

            # Forward pass
            outputs = model(inputs)

            breakpoint()
            # Calculate the loss
            loss = criterion(outputs, inputs)

            # Backpropagation and optimization
            loss.backward()
            optim.step()

            # Update the total loss
            batch_loss.append(loss.item())
            print(f"Batch {batch_idx} loss: {loss.item()}")

    # Calculate the average loss for this epoch
    avg_loss = sum(batch_loss) / len(batch_loss)
    epoch_loss.append(avg_loss)
    # Print the progress
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

print("Training finished!")
