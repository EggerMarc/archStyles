import torch
import torchvision
from image import ArchImages, train_loader

# from torch.utils.data import DataLoader
from encoder import Encoder
from decoder import Decoder


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
        self.encoder = torch.nn.Sequential()
        out_channels = 16
        kernel_size = 5
        self.encoder.add_module(
            f"Convolution 0",
            torch.nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=5),
        )
        self.encoder.add_module(f"Activation 0", torch.nn.ReLU())
        for k in range(K - 1):
            input_channels = out_channels
            out_channels = out_channels * 2
            self.encoder.add_module(
                f"Convolution {k + 1}",
                torch.nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                ),
            )
            self.encoder.add_module(f"Relu Activation {k + 1}", torch.nn.ReLU())

        self.encoder.add_module(f"Flatten", torch.nn.Flatten())

        size = out_channels * (256 - K * (kernel_size - 1)) ** 2

        self.encoder.add_module(
            f"Linear", torch.nn.Linear(in_features=size, out_features=n_features)
        )
        self.encoder.add_module(f"Softmax", torch.nn.Softmax())
        # N, out_channels * 2 * K, 256 - K * (kernel_size - 1), 256 - K * (kernel_size - 1)

        input_channels = n_features
        out_channels = size
        self.decoder = torch.nn.Sequential()
        for k in range(K - 1):
            input_channels = out_channels
            out_channels = 2 * out_channels
            self.decoder.add_module(
                f"ConvTranspose 2D {k + 1}",
                torch.nn.ConvTranspose2d(in_channels=input_channels, out_channels=...),
            )

    def forward(self, X):
        features = self.encoder(X)
        breakpoint()
        # out = self.decoder(features)
        return
        # return out


class Model_0_3(torch.nn.Module):
    def __init__(self, K, n_features):
        super().__init__()

        self.encoder = Encoder(K, n_features=n_features)
        self.decoder = Decoder(K, n_features=n_features)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


if __name__ == "__main__":
    train_loader = train_loader()
    criterion = torch.nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model_0_3(K=5, n_features=100)
    # model = Model_0_2(input_channels=3, K=5, N=100).to(device=device)

    optim = torch.optim.Adam(params=model.parameters(), lr=1e-10)

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
            # Calculate the loss
            loss = criterion(outputs, inputs)

            # Backpropagation and optimization
            loss.backward()
            optim.step()

            # Update the total loss
            batch_loss.append(loss.item())
            # print(f"Batch {batch_idx} loss: {loss.item()}")

    # Calculate the average loss for this epoch
    avg_loss = sum(batch_loss) / len(batch_loss)
    epoch_loss.append(avg_loss)
    # Print the progress
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

print("Training finished!")
