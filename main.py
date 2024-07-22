import os

import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer

import random

from get_dataset import CreateDataset # custom file
from train import train # custom file

seed = 137
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

os.makedirs('dataset', exist_ok = True)
os.makedirs('indexes', exist_ok = True)
os.makedirs('saved_models', exist_ok = True)

dataset_name = "Dataset" # the dataset name
df = pd.read_csv(f"dataset/{dataset_name}.csv") # loading csv file

TEXT = "Text" # the text column name
SENTIMENT = "Sentiment" # the sentiment column name

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5

INDEX_DIRECTORY = "indexes"
model_save_directory = "saved_models"
saved_model_name = "finbert"

model_name = "ProsusAI/finbert"
num_classes = 3

tokenizer = AutoTokenizer.from_pretrained(model_name)

torch_dataset = CreateDataset(df, source_column = TEXT, target_column = SENTIMENT, max_length = MAX_LENGTH, tokenizer = tokenizer, seed = seed)

train(dataset = torch_dataset, index_directory = INDEX_DIRECTORY, batch_size = BATCH_SIZE, 
            EPOCHS = EPOCHS, MAX_LENGTH = MAX_LENGTH, LEARNING_RATE = LEARNING_RATE, 
            original_dataset = df, model_name = model_name, num_classes = num_classes,
            SAVE_PATH = model_save_directory, name = saved_model_name)