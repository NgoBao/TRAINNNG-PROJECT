"""
MNIST Dataset Loading Example for Deep Learning Projects
=====================================================

This script demonstrates how to load and preprocess the MNIST dataset using PyTorch.
MNIST is a database of handwritten digits (0-9) commonly used for training various
image processing and machine learning systems.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def dataloader(train_dataset, test_dataset, batch_size=128):
    """
    Creates DataLoader objects for both training and testing datasets.
    """
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def load_data():
    """
    Loads and preprocesses the MNIST dataset.
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=False,
        download=True,
        transform=transform
    )

    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))

    return dataloader(train_dataset, test_dataset, batch_size=128)