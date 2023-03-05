import argparse
import torchvision

dataset_list = [
    'ImageNet', 
    'CIFAR10',
    'CIFAR100',
    'MNIST',
    ]

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, choices=dataset_list, required=True)
parser.add_argument('-o', '--output', type=str, default="./data")

def download_data(dataset_name, output_folder):
    if dataset_name == 'ImageNet':
        torchvision.datasets.ImageNet(output_folder, download=True, train=True)
        torchvision.datasets.ImageNet(output_folder, download=True, train=False)
    elif dataset_name == 'CIFAR10':
        torchvision.datasets.CIFAR10(output_folder, download=True, train=True)
        torchvision.datasets.CIFAR10(output_folder, download=True, train=False)
    elif dataset_name == 'CIFAR100':
        torchvision.datasets.CIFAR100(output_folder, download=True, train=True)
        torchvision.datasets.CIFAR100(output_folder, download=True, train=False)
    elif dataset_name == 'MNIST':
        torchvision.datasets.MNIST(output_folder, download=True, train=True)
        torchvision.datasets.MNIST(output_folder, download=True, train=False)

if __name__ == "__main__":
    args = parser.parse_args()
    download_data(args.dataset, args.output)