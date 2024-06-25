import torch
import re
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

df = pd.read_csv('./sms.csv')
print("Beérkezett SMS-ek száma", len(df))
df.head()

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

def unique_word_count():
    # count the unique words in the dataset
    words = []
    for i in range(len(df)):
        words += df['text'][i].split()
    return len(Counter(words))

def sentiment_count():
    raise NotImplementedError("This function has not been implemented yet.")