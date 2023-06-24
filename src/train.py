import os
import argparse
import random

import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("--width", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--model-name", type=str)
parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction)
parser.add_argument('--experiment', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

class Config:
    batch_size   = args.batch_size or 32
    epochs       = args.epochs or 5
    image_heigth = args.height or 224
    image_width  = args.width or 224
    model_name   = args.model_name or "unnamed_model"
    num_workers  = os.cpu_count()
    device       = "cuda" if torch.cuda.is_available() else "cpu"

def augmented_transform(transform=None):
    return T.Compose([
        T.TrivialAugmentWide(),
        transform,
    ])

def to_dataloader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=shuffle,
        num_workers=Config.num_workers,
        pin_memory=True
    )

