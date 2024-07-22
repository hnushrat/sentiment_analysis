import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

import random

########################
seed = 42
random.seed(seed)
np.random.seed(seed)
########################

TEXT = "News Headlines"
SENTIMENT = "Market Sentiment"
SAVE_PATH = "indexes"
dataset_name = "Final_Sentiment_analysis_news_data_v1"

indices = []
df = pd.read_csv(f"dataset/{dataset_name}.csv")

for fold, (train_idx, test_idx) in enumerate(StratifiedKFold(5, shuffle = True).split(df, y = df[SENTIMENT].values)):
    print(f"Fold: {fold+1}\n")
    
    fold = f"Fold_{fold+1}"
    
    ###### OPTIONAL ######
    # df.loc[train_idx][[TEXT, SENTIMENT]].to_csv(f"dataset/{fold}_{dataset_name}_train.csv", index = False)
    # df.loc[test_idx][[TEXT, SENTIMENT]].to_csv(f"dataset/{fold}_{dataset_name}_test.csv", index = False)
    ###### OPTIONAL ######
    
    np.save(f"{SAVE_PATH}/{fold}_train_idx.npy", train_idx)
    np.save(f"{SAVE_PATH}/{fold}_test_idx.npy", test_idx)