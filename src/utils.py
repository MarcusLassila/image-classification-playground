import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch

def save_model(model, target_dir, model_name):
    '''Save model (state dict) to <target_dir>/<model_name>.pth'''
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path / (model_name + '.pth')
    print(f"[info] Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def load_model(model, save_path):
    model.load_state_dict(torch.load(f=save_path))
    model.eval()
    return model

def plot_random_samples(dataset, dim=4):
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, dim ** 2 + 1):
        image, label = random.choice(dataset)
        fig.add_subplot(dim, dim, i)
        if isinstance(image, Image.Image):
            plt.imshow(image)
        else:
            plt.imshow(image.permute(1, 2, 0))
        plt.title(label)
        plt.axis("off")
    plt.show()

def plot_training_data(training_data, criterion="criterion"):
    epochs = range(len(training_data['train_loss']))
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_data['train_loss'], label="train_loss")
    plt.plot(epochs, training_data['valid_loss'], label="valid_loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_data['train_crit'], label=f"train_{criterion}")
    plt.plot(epochs, training_data['valid_crit'], label=f"valid_{criterion}")
    plt.title(criterion.capitalize())
    plt.xlabel("epochs")
    plt.yticks(np.arange(0, 1.05, step=0.05))
    plt.grid()
    plt.legend()

    plt.show()
