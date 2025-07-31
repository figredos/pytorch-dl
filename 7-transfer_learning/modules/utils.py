"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
from pathlib import Path


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
