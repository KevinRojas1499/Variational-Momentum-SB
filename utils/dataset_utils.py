import click
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

@click.command()
@click.option('--dataset', type=click.Choice(['mnist', 'cifar', 'fashion']), default='cifar')
def download_dataset(dataset):
    # Create a simple object with dot notation
    opts = type('', (), {'dataset': dataset})() #Small hack to avoid import
    get_dataset(opts)

def get_dataset(opts):
    # Returns dataset, data_shape, num_labels
    ds_name = opts.dataset
    if ds_name == 'mnist':
        dataset = MNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
        return dataset, [1,28,28], 10
    elif ds_name == 'fashion':
        dataset = FashionMNIST('.', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
        return dataset, [1,28,28], 10
    elif ds_name == 'cifar':
        dataset= CIFAR10('.', train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(p=0.5)]), download=True)
        return dataset, [3,32,32], 10
    else:
        print('Dataset is not implemented')

if __name__ == '__main__':
    download_dataset()