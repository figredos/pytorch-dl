"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch

from torch.utils.tensorboard import SummaryWriter

import os
from pathlib import Path
from datetime import datetime


def save_model(model: torch.nn.Module, target_dir: str | Path, model_name: str) -> None:
    """
    Saves a PyTorch model to a target directory.

    Args:
        model (torch.nn.Module): A PyTorch model to save.
        target_dir (str | Path): Directory in which the model should be saved.
        model_name (str): The filename for the saved model. Should include either
        ".pth" or ".pt" as the file extension
    """
    target_dir_path = target_dir
    if not isinstance(target_dir_path, Path):
        target_dir_path = Path(target_dir)

    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'."
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def create_writer(
    experiment_name: str,
    model_name: str,
    extra: str | None = None,
) -> SummaryWriter:
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance tracking to a specific directory.

    Args:
        experiment_name (str): Name of the experiment
        model_name (str): Name of the model
        extra (str | None = None): Extra pertinent information

    Returns:
        torch.utils.tensorboard.`SummaryWriter`: Instance of `SummaryWriter` with `log_dir` set as a combination of the parameters.
    """

    log_dir = os.path.join("runs", experiment_name, model_name)

    if extra:
        log_dir = os.path.join(log_dir, extra)

    return SummaryWriter(log_dir=log_dir)
