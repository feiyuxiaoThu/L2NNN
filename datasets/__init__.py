import torch
from torch.utils.data import Sampler, DataLoader
import torchvision
from torchvision import datasets as ds
from torchvision import transforms as T


class IndiceSampler(Sampler):
    r'''Sample from a dataset by indice.

    Args:
        indice (list): indice of elements to sample.

    Example:
        >>> import torchvision
        >>> ds = torchvision.datasets.mnist.MNIST(root='./MNIST_data', download=True, train=True)
        >>> sampler = IndiceSampler(range(10))
        >>> loader = torch.utils.data.DataLoader(ds,
        ...     batch_size=5,
        ...     sampler=sampler,
        ...     shuffle=False,
        ...     num_workers=2
        ...     )
        >>> len(loader)
        2
    '''

    def __init__(self, indice):
        self.indice = indice

    def __iter__(self):
        return self.indice

    def __len__(self):
        return len(self.indice)


def get_data(config):

    def trans(im_size): return T.Compose([T.Resize(im_size), T.ToTensor()])
    def aug_trans(im_size): return T.Compose([T.RandomCrop(im_size, 4), T.ToTensor()])
    if config.dataset == 'mnist':
        train_ds = ds.MNIST(root='./data/MNIST_data', train=True, transform=trans(config.im_size), download=True)
        test_ds = ds.MNIST(root='./data/MNIST_data', train=False, transform=trans(config.im_size), download=True)
        classes = list(range(10))
    elif config.dataset == 'cifar10':
        train_ds = ds.CIFAR10(root='./data/CIFAR10_data', train=True, transform=trans(config.im_size), download=True)
        test_ds = ds.CIFAR10(root='./data/CIFAR10_data', train=False, transform=trans(config.im_size), download=True)
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif config.dataset == 'FashionMNIST':
        classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.test_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    return train_ds, test_ds, train_loader, test_loader, classes
