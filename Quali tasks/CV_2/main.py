import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as models

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models.efficientnet import EfficientNet_B2_Weights
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = {0: 'butterfly',
           1: 'cat',
           2: 'cow',
           3: 'dog',
           4: 'elephant',
           5: 'hen',
           6: 'horse',
           7: 'sheep',
           8: 'spider',
           9: 'squirrel'}

model = models.efficientnet_b2(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(classes)),
    nn.Softmax(dim=1)
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

model.load_state_dict(torch.load('jános_modell.pth', map_location=device))


class PhotoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


data_dir = 'állatok'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = PhotoDataset(data_dir, transform=transform)

animal_counts = {class_name: 0 for class_name in classes.values()}

# Számoljuk meg az egyes állatokat
for filename in tqdm(os.listdir(data_dir)):
    image_path = os.path.join(data_dir, filename)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Hozzáadjuk a batch dimenziót

    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

        predicted_class = classes[predicted.item()]
        animal_counts[predicted_class] += 1

# Eredmények kiíratása
for animal, count in animal_counts.items():
    print(f'{animal} = {count}')