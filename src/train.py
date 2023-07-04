import data_setup
import engine
import pretrained_models
import utils
from infer import infer

import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchmetrics import Accuracy, F1Score

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--model-name", default="unnamed_model", type=str)
parser.add_argument("--save-path", default="", type=str)
parser.add_argument('--pretrained', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

class Config:
    batch_size   = args.batch_size
    epochs       = args.epochs
    model_name   = args.model_name
    save_path    = args.save_path
    num_workers  = os.cpu_count()
    device       = "cuda" if torch.cuda.is_available() else "cpu"

def augmented_transform(transform=None):
    return T.Compose([
        T.TrivialAugmentWide(),
        transform,
    ])

def run():

    pure_data = data_setup.CIFAR10(train_transform=T.ToTensor(),
                                   valid_transform=T.ToTensor(),
                                   batch_size=Config.batch_size,
                                   num_workers=Config.num_workers)

    utils.plot_random_samples(pure_data.train_dataset)

    model = pretrained_models.efficient_net(pure_data.num_classes, 'b0')
    data  = data_setup.CIFAR10(train_transform=augmented_transform(model.transforms),
                               valid_transform=model.transforms,
                               batch_size=Config.batch_size,
                               num_workers=Config.num_workers)

    params = [
        {'params': model.features.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
    ]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], total_steps=Config.epochs)
    criterion = Accuracy(task="multiclass", num_classes=data.num_classes)

    training_data = engine.training_loop(model,
                                         data.train_dataloader,
                                         data.valid_dataloader,
                                         loss_fn, optimizer,
                                         criterion,
                                         Config.device,
                                         Config.epochs,
                                         scheduler=scheduler)

    utils.plot_training_data(training_data, criterion="Accuracy")
    if Config.save_path:
        utils.save_model(model, target_dir=f'{Config.save_path}/models', model_name=Config.model_name)
    #infer(model, "data/flowers-102/jpg/image_00020.jpg")

if __name__ == '__main__':
    run()
