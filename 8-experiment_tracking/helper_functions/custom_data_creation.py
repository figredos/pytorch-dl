from typing import Tuple, List, Dict, Any

import shutil
import random
import pathlib

from torch.utils.data import Dataset

import torchvision.datasets as datasets


def download_data(
    target_dir: str, download: bool = True
) -> Tuple[Dataset, Dataset, List[str]]:
    """
    Downloads Food101 train and test datasets from `torchvision.datasets`.

    Args:
        target_dir (str): Directory to download data to.
        download (bool): Bool value to determine whether to download the dataset or not.

    Returns:
        `Tuple[Dataset, Dataset, List[str]]`: A tuple with the train and test datasets, and a list of class names.
    """
    data_dir = pathlib.Path(target_dir)

    train_data = datasets.Food101(root=data_dir, split="train", download=download)
    test_data = datasets.Food101(root=data_dir, split="test", download=download)

    class_names = train_data.classes

    return train_data, test_data, class_names


def get_subset(
    data_dir: pathlib.Path,
    image_path: pathlib.Path,
    data_splits: List[str],
    target_classes: List[str],
    sample: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """
    Selects a subset of classes from original dataset, based on select `target_classes`.

    Args:
        data_dir (pathlib.Path): Path for data to be split.
        image_path (pathlib.Path): Path for image data.
        data_splits (List[str]): Names of created splits.
        target_classes (List[str]): Names of classes to select.
        sample (float = 0.1): Amount of data to sample. 0.1 (10%) by default.
        seed (int = 42): Seed for random sampling. 42 by default.

    Returns:
        `Dict[str, List[Any]]`: Dictionary of equally split labels from `data_splits` contents, and sampled based on `sample` value.
    """
    random.seed(seed)
    label_splits = {}

    for data_split in data_splits:
        print(f"[INFO] Creating image split for: {data_split}.")
        label_path = data_dir / "food-101" / "meta" / f"{data_split}.txt"

        print(data_split)

        with open(label_path, "r") as f:
            labels = [
                line.strip("\n")
                for line in f.readlines()
                if line.split("/")[0] in target_classes
            ]

        number_to_sample = round(sample * len(labels))
        sampled_images = random.sample(labels, k=number_to_sample)

        image_paths = [
            pathlib.Path(str(image_path / sampled_image) + ".jpg")
            for sampled_image in sampled_images
        ]
        label_splits[data_split] = image_paths

    return label_splits


def move_and_zip_data(target_dir_name: str, label_splits: Dict[str, List[Any]]) -> None:
    """
    Moves select images into a target directory based on the `label_splits` dictionary.

    Args:
        target_dir_name (str): Name for target directory.
        label_splits (Dict[str, List[Any]]): Dictionary of split image labels.
    """

    target_dir = pathlib.Path(target_dir_name)

    target_dir.mkdir(parents=True, exist_ok=True)

    for image_split in label_splits.keys():
        for image_path in label_splits[str(image_split)]:
            print(image_path)
            dest_dir = (
                target_dir / image_split / image_path.parent.stem / image_path.name
            )

            print(dest_dir)
            if not dest_dir.parent.is_dir():
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
            print(image_path)
            shutil.copy2(image_path, dest_dir)

    shutil.make_archive(target_dir_name, format="zip", root_dir=target_dir)
