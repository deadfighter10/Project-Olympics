import os
import glob
import imageio
import random, shutil
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import librosa
import librosa.display
from zipfile import ZipFile
import requests

fname = "music.zip"

sample_path = 'Data/genres_original/jazz/jazz.00000.wav'

# if you want to listen to the audio, uncomment below.
display.Audio(sample_path)

y, sample_rate = librosa.load(sample_path)

print('y:', y, '\n')
print('y shape:', np.shape(y), '\n')
print('Sample rate (KHz):', sample_rate, '\n')
print(f'Length of audio: {np.shape(y)[0]/sample_rate}')


D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
DB = librosa.amplitude_to_db(D, ref=np.max)

# Convert sound wave to mel spectrogram.

y, sr = librosa.load(sample_path)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64) #(y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)

img_path = 'Data/images_original/jazz/jazz00000.png'
img = imageio.v2.imread(img_path)

spectrograms_dir = "Data/images_original/"
folder_names = ['Data/train/', 'Data/test/']
train_dir = folder_names[0]
test_dir = folder_names[1]

for f in folder_names:
    if os.path.exists(f):
        shutil.rmtree(f)
        os.mkdir(f)
    else:
        os.mkdir(f)

# Loop over all genres.


genres = list(os.listdir(spectrograms_dir))
for g in genres:
    # find all images & split in train and test sets
    src_file_paths= []
    for im in glob.glob(os.path.join(spectrograms_dir, f'{g}',"*.png"), recursive=True):
        src_file_paths.append(im)
    random.shuffle(src_file_paths)
    test_files = src_file_paths[0:20]
    train_files = src_file_paths[20:]

    #  make destination folders for train and test images
    for f in folder_names:
        if not os.path.exists(os.path.join(f + f"{g}")):
            os.mkdir(os.path.join(f + f"{g}"))

    # copy training and testing images over
    for f in train_files:
        shutil.copy(f, os.path.join(os.path.join(train_dir + f"{g}") + '/',os.path.split(f)[1]))
    for f in test_files:
        shutil.copy(f, os.path.join(os.path.join(test_dir + f"{g}") + '/',os.path.split(f)[1]))

# Data loading.

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.ToTensor(),
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=25, shuffle=True, num_workers=0)

test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.ToTensor(),
    ]))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=25, shuffle=False, num_workers=0)

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
      print("WARNING: For this notebook to perform best, "
            "if possible, in the menu under `Runtime` -> "
            "`Change runtime type.`  select `GPU` ")
  else:
      print("GPU is enabled in this notebook.")

  return device

device = set_device()