"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""

import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(
    train_dir: str | Path,
    test_dir: str | Path,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
):
    """
    Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns them into PyTorch Datasets
    and then into PyTorch DataLoaders.

    Args;
        train_dir (str | Path): Path to training directory.
        test_dir (str | Path): Path to testing directory.
        train_transform (transforms.Compose): `torchvision` transforms to perform on training data data.
        test_transform (transforms.Compose): `torchvision` transforms to perform on testing data.
        batch_size (int): Number of samples per batch in each of the DataLoaders.

    Returns:
        `Tuple[DataLoader,DataLoader,str]`: A tuple of `(train_dataloader, test_dataloader, class_names)`,
        where `class_names` is a list of the target classes.
    """

    # Using ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Getting class names
    class_names = train_data.classes

    # Turn images into dataloaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, test_dataloader, class_names
