import numpy as np

import pandas as pd

from datasets import Dataset

import torch
from torch.utils.data import Dataset

import random

class CreateDataset(Dataset):
        '''        
        df: the training dataframe
        source_column : the name of source text column in the dataframe
        target_columns : the name of target column in the dataframe
        
        returns tokens, attention masks and one-hot encoded label in the order -> (Positive, Negative, Neutral)
        '''
        
        def __init__(self, df, source_column, target_column, max_length, tokenizer, seed = None):

            if not (seed == None):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            self.df = df        
            self.source_text = self.df[source_column].values
            self.target = (pd.get_dummies(self.df[target_column])[["Positive","Negative","Neutral"]]).values # one-hot encoded
            self.MAX_LENGTH = max_length
            self.tokenizer = tokenizer
            
        def __len__(self): # returns size of the dataset
            return len(self.df)
        
        def __getitem__(self, index):
            '''
            __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
            target values we created in __init__
            
            '''
            source_text = self.source_text[index]
            target_text = self.target[index]
            
            temp = self.tokenizer.encode_plus(source_text, truncation = True, 
                                              padding = "max_length", max_length = self.MAX_LENGTH,
                                              return_tensors = 'pt')

            return temp["input_ids"], temp["attention_mask"], torch.tensor(target_text, dtype = torch.float)