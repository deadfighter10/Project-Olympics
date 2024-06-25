import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt
import multiprocessing
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'A random seed értéke: {seed}')


# In case that `DataLoader` is used
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# inform the user if the notebook uses GPU or CPU.

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


set_seed(seed=2024)
device = set_device()


def get_dataloaders(dataset_name, batch_size=256, augmentation=False):
    normalization_data = {'CIFAR100': ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
                          'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))}

    transform_train = transforms.Compose([])
    if augmentation:
        transform_train.transforms.append(transforms.RandomCrop(32, padding=4))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())

    transform_train.transforms.append(transforms.ToTensor())
    transform_train.transforms.append(transforms.Normalize(mean=normalization_data[dataset_name][0],
                                                           std=normalization_data[dataset_name][1]))

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=normalization_data[dataset_name][0],
                                                              std=normalization_data[dataset_name][1])])
    if dataset_name == 'CIFAR100':
        dataclass = torchvision.datasets.CIFAR100
    elif dataset_name == 'CIFAR10':
        dataclass = torchvision.datasets.CIFAR10

    trainset = dataclass(root=f'./{dataset_name}', train=True, download=True, transform=transform_train)
    testset = dataclass(root=f'./{dataset_name}', train=False, download=True, transform=transform_test)

    print(f"Objektum: {type(trainset)}")
    print(f"Tanító adatok shape-je: {trainset.data.shape}")
    print(f"Teszt adatok shape-je: {testset.data.shape}")
    print(f"Az osztályok száma: {np.unique(trainset.targets).shape[0]}")

    #num_workers = multiprocessing.cpu_count()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def evaluate_model(model, dataloader, device='cuda'):
    model.eval()  # Állítsuk a modellt kiértékelési módba
    criterion = nn.CrossEntropyLoss()  # Definiáljuk a keresztentrópia veszteségfüggvényt

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Kikapcsoljuk a gradiens számolást
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = correct_predictions / total_samples

    return avg_loss, avg_accuracy

if __name__ == '__main__':
    cifar100_trainloader, cifar100_testloader = get_dataloaders('CIFAR100')
    cifar10_trainloader, cifar10_testloader = get_dataloaders('CIFAR10')

    cifar10_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    cifar100_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)

    cifar10_embeddings, labels = torch.load(
        r'C:\Users\David\PycharmProjects\Project-Olympics\content\cifar10_embeddings_and_labels.pt',
        map_location=torch.device('cuda'))
    avg_loss, avg_accuracy = evaluate_model(cifar10_model, cifar10_testloader, "cpu")
    print(f'Average Loss: {avg_loss}')
    print(f'Average Accuracy: {avg_accuracy}')