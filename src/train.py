from data_setup import Flowers102DataSetup
import engine
import pretrained_models
import utils

import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchmetrics import F1Score

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--model-name", type=str)
parser.add_argument("--save-path", type=str)
parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction)
parser.add_argument('--experiment', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

class Config:
    batch_size   = args.batch_size or 32
    epochs       = args.epochs or 5
    model_name   = args.model_name or "unnamed_model"
    save_path    = args.save_path or None
    num_classes  = 102
    num_workers  = os.cpu_count()
    device       = "cuda" if torch.cuda.is_available() else "cpu"

def augmented_transform(transform=None):
    return T.Compose([
        T.TrivialAugmentWide(),
        transform,
    ])

def run():
    model = pretrained_models.efficient_net(102, 'b0')
    params = [
        {'params': model.features.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
    ]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], total_steps=Config.epochs)
    criterion = F1Score('multiclass', num_classes=Config.num_classes)
    flowers102 = Flowers102DataSetup(train_transform=augmented_transform(model.transforms),
                                     test_transform=model.transforms,
                                     batch_size=Config.batch_size,
                                     num_workers=Config.num_workers)

    training_data = engine.training_loop(model,
                                         flowers102.train_dataloader,
                                         flowers102.valid_dataloader,
                                         loss_fn, optimizer,
                                         criterion,
                                         Config.device,
                                         Config.epochs,
                                         scheduler=scheduler)

    utils.plot_training_data(training_data, criterion="F1-Score")
    if Config.save_path:
        utils.save_model(model, target_dir=f'{Config.save_path}/models', model_name=Config.model_name)

if __name__ == '__main__':
    run()
