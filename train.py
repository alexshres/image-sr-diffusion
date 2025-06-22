import unet.py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_FMNIST():
    transform = transforms.Compose([transforms.ToTensor()])

    full_dataset = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
            )

    train_size = int(0.9*len(full_dataset))
    val_size = len(full_dataset)-train_size


    train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size]
            )

    test_dataset = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
            )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader







