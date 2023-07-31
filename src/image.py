import pandas as pd
import os
from skimage import io
from PIL import Image
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset


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
