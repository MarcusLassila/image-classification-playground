import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(training_data, criterion="criterion"):
    epochs = range(len(training_data['train_loss']))
    plt.figure(figsize=(16, 8))

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
