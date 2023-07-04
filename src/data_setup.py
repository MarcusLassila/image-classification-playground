from torch.utils.data import DataLoader, Subset
import torchvision

class Flowers102:
    def __init__(self, train_transform=None, test_transform=None, batch_size=1, num_workers=0):
        self.num_classes      = 102
        self.train_dataset    = torchvision.datasets.Flowers102(root='./data', split='train', transform=train_transform, download=True)
        self.valid_dataset    = torchvision.datasets.Flowers102(root='./data', split='val', transform=test_transform, download=True)
        self.test_dataset     = torchvision.datasets.Flowers102(root='./data', split='test', transform=test_transform, download=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_dataloader  = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f'[info] (train size, valid size, test size) = {(len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))}')

class CIFAR10:
    def __init__(self, train_transform=None, valid_transform=None, batch_size=1, num_workers=0):
        self.num_classes      = 10
        self.train_dataset    = torchvision.datasets.CIFAR10(root='/data', train=True, transform=train_transform, download=True)
        self.valid_dataset    = torchvision.datasets.CIFAR10(root='/data', train=False, transform=valid_transform, download=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f'[info] (train size, valid size) = {(len(self.train_dataset), len(self.valid_dataset))}')

if __name__ == '__main__':
    cifar100 = CIFAR10()
