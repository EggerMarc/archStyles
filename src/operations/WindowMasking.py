import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowMasking(nn.Module):
    """
    Source: https://discuss.pytorch.org/t/masked-sliding-window-tensor/180517/4
    """

    def __init__(self, kernel_size: int = 16, stride: int = 16, device: str = "cpu"):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        mask = torch.ones_like(x, device=self.device)
        mask = F.unfold(mask, kernel_size=self.kernel_size, stride=self.stride)

        N, chw, p = mask.shape

        mask = mask.unsqueeze(3).expand(N, chw, p, p).clone()

        N, c, h, w = x.shape

        mask = mask.reshape(N, c, self.kernel_size, self.kernel_size, p, p)
        mask = mask.permute(0, 4, 1, 2, 3, 5).reshape(
            N * p * c, self.kernel_size**2, p
        )
        mask = F.fold(mask, (h, w), kernel_size=self.kernel_size, stride=self.stride)
        mask = mask.reshape(N, p, c, h, w)

        out = x.unsqueeze(1).expand(N, p, c, h, w).clone()
        out[~mask.bool()] = 0

        return out


if __name__ == "__main__":
    x = torch.randn((16, 56, 56, 96))

    masker = WindowMasking(16, 16)
    masked = masker(x)
