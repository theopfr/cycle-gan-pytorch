
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import List
from PIL import Image
import os
import random


class cycleGanDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, image_size: int) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size

        self.images_a = os.listdir(f"../datasets/{self.dataset_path}/trainA/")
        self.images_b = os.listdir(f"../datasets/{self.dataset_path}/trainB/")

        np.random.shuffle(self.images_a)
        np.random.shuffle(self.images_b)

    def __getitem__(self, _: int) -> torch.Tensor:
        idx_a = random.randint(0, len(self.images_a) - 1)
        idx_b = random.randint(0, len(self.images_b) - 1)

        image_a = np.array(Image.open(f"../datasets/{self.dataset_path}/trainA/" + self.images_a[idx_a]).convert("RGB"))
        image_b = np.array(Image.open(f"../datasets/{self.dataset_path}/trainB/" + self.images_b[idx_b]).convert("RGB"))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(size=self.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transform(image_a), transform(image_b)

    def __len__(self):
        return min([len(self.images_a), len(self.images_b)]) - 1

