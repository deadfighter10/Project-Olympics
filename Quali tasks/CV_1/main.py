import os

import easyocr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

plt.style.use('ggplot')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reader = easyocr.Reader(['en'])


class MountainDataset(Dataset):
    def __init__(self, directory, transforms=None):
        self.img_dir = directory
        self.img_list = os.listdir(directory)
        self.transform = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor()
])

full_dataset = MountainDataset(directory='hegyek', transforms=transform)

full_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)

to_pil = ToPILImage()

for batch in full_loader:
    img_tensor = batch.squeeze(0)
    img = to_pil(img_tensor)
    img_np = np.array(img)
    result = reader.readtext(img_np)
    for (bbox, text, prob) in result:
        if float(prob) > 0.5 and len(str(text)) > 1:
            print(f"Detected text: {text} (Confidence: {prob:.2f})")
    #print("===================")
