"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse

import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils

# Setup parser
parser = argparse.ArgumentParser(description="Train a TinyVGG model.")
parser.add_argument(
    "-ne",
    "--num_epochs",
    type=int,
    default=5,
    help="Number of epochs to train model.",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=32,
    help="Number of batches to group data.",
)
parser.add_argument(
    "-hu",
    "--hidden_units",
    type=int,
    default=10,
    help="Number of hidden units in model architecture.",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.001,
    help="Value for model's learning rate.",
)
parser.add_argument(
    "-mn",
    "--model_name",
    type=str,
    default="script_mode.pth",
    help="Name for saved model.",
)

# Getting args
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
MODEL_NAME = args.model_name


if MODEL_NAME[-3:] != "pth" and MODEL_NAME[-2:] != 'pt':
    MODEL_NAME += ".pth"

# Setup directories
train_dir = "../data/pizza_steak_sushi/train"
test_dir = "../data/pizza_steak_sushi/test"

# Setup target device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "mps" if torch.mps.is_available() else "cpu"

# Creating transforms
data_transform = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ]
)
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Creating DataLoaders with ,helps from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=train_transform,
    test_transform=data_transform,
    batch_size=BATCH_SIZE,
)

# Creating model with ,help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
).to(device)

# Setting up loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Start training with ,help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
)

# Saving model with ,help from utils.py
utils.save_model(model=model, target_dir="../models", model_name=MODEL_NAME)
