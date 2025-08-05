"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode then runs through all of the required training steps
    (forward pass, loss calculation, optimizer step).

    Args:
        model (torch.nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        device (str): A target device's name to compute on.

    Returns:
        `Tuple[float,float]`: Tuple of training loss and training accuracy metrics.
    """

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: str | torch.device,
) -> Tuple[float, float]:
    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to eval mode then runs through forward pass on test dataset.

    Args:
        model (torch.nn.Module): A PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        loss_fn (torch.nn.Module): A PyTorch loss function to calculate loss on test data.
        device (str): A target device's name to compute on.

    Returns:
        `Tuple[float,float]`: Tuple of testing loss and testing accuracy metrics.
    """

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    writer: SummaryWriter | None = None,
    epochs: int = 5,
    device: str | torch.device = "cpu",
) -> Dict[str, List[float]]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode then runs through all of the required training steps
    (forward pass, loss calculation, optimizer step).

    Args:
        model (torch.nn.Module): A PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        writer (torch.utils.tensorboard.writer.SummaryWriter | None): Instance of Summary to track experiments.
        epochs (int): Number of epochs to train on.
        device (str): A target device's name to compute on.

    Returns:
        `Dict[str,List[float]]`: Dictionary of training and testing loss, as well as training and testing
        accuracy metrics. Each metric has a value in a list for each epoch.

    """

    results = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"""
======================================      
Epoch:                       |   {epoch + 1}   |  
======================================      
Train Loss:                  | {train_loss:.3f} |

Train Accuracy:              | {train_acc:.3f} |            
--------------------------------------
Test Loss:                   | {test_loss:.3f} |

Test Accuracy:               | {test_acc:.3f} |
              """
        )

        results["train_loss"].append(
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        results["train_acc"].append(
            train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc
        )
        results["test_loss"].append(
            test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        )
        results["test_acc"].append(
            test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc
        )

        if writer is not None:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )
            writer.add_graph(
                model=model, input_to_model=torch.randn(32, 3, 224, 225).to(device)
            )

    if writer is not None:
        writer.close()

    return results
