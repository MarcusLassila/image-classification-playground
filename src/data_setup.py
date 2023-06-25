from torch.utils.data import DataLoader, Subset
import torchvision

class Flowers102DataSetup:

    def __init__(self, train_transform=None, test_transform=None, batch_size=1, num_workers=0):
        self.train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', transform=train_transform, download=True)
        self.valid_dataset = torchvision.datasets.Flowers102(root='./data', split='val', transform=None, download=True)
        self.test_dataset  = torchvision.datasets.Flowers102(root='./data', split='test', transform=test_transform, download=True)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.valid_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        self.test_dataloader  = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == '__main__':
    pass
