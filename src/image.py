import pandas as pd
import os
from skimage import io
from PIL import Image
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset

"""

TODO:
1. Fix inconsistent sizes after compose. E.g. Error line: 
*** RuntimeError: stack expects each tensor to be equal size, 
but got [3, 256, 256] at entry 0 and [4, 256, 256] at entry 3
"""


class ArchImages(Dataset):
    def __init__(self, root_dir, transform=None, device="cpu"):
        self.root_dir = root_dir
        self.transform = transform
        self.label = os.listdir(self.root_dir)
        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # 1. Get photo from path
        img_path = os.path.join(self.root_dir, f"{index + 1}.jpg")
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            return self.__getitem__(index - 1)
        # 2. Transform photo (crop and center)
        if self.transform:
            return self.transform(img)
        # 3. Return photo
        return img


def train_loader(resize=256, batch_size=16, shuffle=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ArchImages(
        "../images",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(int(resize * (3 / 2)), antialias=True),
                torchvision.transforms.CenterCrop(resize)),
                torchvision.transforms.
            ]
        ),
        device=device,
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    breakpoint()
    ArchImages("../images")
