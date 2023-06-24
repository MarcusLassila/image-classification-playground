import matplotlib.pyplot as plt
import random
import torchvision

class Flowers102DataSetup:
    
    def __init__(self, train_transform=None, test_transform=None):
        self.train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', transform=train_transform, download=True)
        self.valid_dataset = torchvision.datasets.Flowers102(root='./data', split='val', transform=test_transform, download=True)
        self.test_dataset  = torchvision.datasets.Flowers102(root='./data', split='test', transform=test_transform, download=True)

    def plot_random_samples(self, dim=4):
        fig = plt.figure(figsize=(10, 10))
        for i in range(1, dim ** 2 + 1):
            image, label = random.choice(self.train_dataset)
            fig.add_subplot(dim, dim, i)
            plt.imshow(image)
            plt.title(label)
            plt.axis("off")
        plt.show()

if __name__ == '__main__':
    flowers102 = Flowers102DataSetup()
    flowers102.plot_random_samples()
