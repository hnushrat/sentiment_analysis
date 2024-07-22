import numpy as np

import torch
from torch import nn

from transformers import AutoModel

import random

class bert(nn.Module):
    def __init__(self, name = None, num_classes = None, seed = None):
        super(bert, self).__init__()
        
        if not (seed == None):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.bert = AutoModel.from_pretrained(name)
        self.linear1 = nn.Linear(768, num_classes)
        
        
                
    def forward(self, input_ids = None, attention_mask = None):
        
        op = self.bert(input_ids = input_ids, attention_mask = attention_mask)# sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(op[0][:, 0, :]) # take the embeddings of the CLS token
        
        return linear1_output