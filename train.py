import numpy as np

import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm

import random

from models import bert # custom file



seed = 137
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(ytrue, ypred):
    correct = torch.eq(ytrue, ypred).sum().item()
    return (correct/len(ypred))*100

def train(dataset, index_directory, batch_size = None, original_dataset = None, 
                EPOCHS = None, MAX_LENGTH = None, LEARNING_RATE = None, model_name = None, num_classes = None,
                SAVE_PATH = None, name = None):
    
    if EPOCHS == None:
        print("\nNumber of epochs is required, and should be greater than 0.\n")
        exit()
    
    if batch_size == None:
        print("\nBatch size is required.\n")
        exit()
    
    if LEARNING_RATE == None:
        print("\nLearning Rate is required!\n")
        exit()
    
    if MAX_LENGTH == None:
        MAX_LENGTH = 512
        print(f"\nMax Sequence Length not provided, switching to default -> 512\n")

    if model_name == None or num_classes == None:
        print("\nmodel_name or num_classes is not provided.\n")
        exit()
    
    for i in range(5):
        
        print(f"Fold: {i+1}\n")
        
        fold = f"Fold_{i+1}"
        
        train_idx = np.load(f"{index_directory}/Fold_{i+1}_train_idx.npy")
        test_idx = np.load(f"{index_directory}/Fold_{i+1}_test_idx.npy")
        
        ###################### OPTIONAL ########################
        # original_dataset.loc[train_idx][[TEXT, SENTIMENT]].to_csv(f"dataset/{fold}_{dataset_name}_train.csv", index = False)
        # original_dataset.loc[test_idx][[TEXT, SENTIMENT]].to_csv(f"dataset/{fold}_{dataset_name}_test.csv", index = False)
        ###################### OPTIONAL ########################
        
        with open(f"log.txt", 'a') as f:
            f.writelines(f"{fold}:\n\n")
        f.close()
        
        best_acc = -1
        train_acc = -1        

        train_dataloader = DataLoader(dataset, batch_size = batch_size, sampler = torch.utils.data.SubsetRandomSampler(train_idx))
        test_dataloader = DataLoader(dataset, batch_size = batch_size, sampler = torch.utils.data.SubsetRandomSampler(test_idx))
        
        model = bert(name = model_name, num_classes = num_classes, seed = seed)
        model.to(device)

        print(f"Model: {model_name}")
        print(f"\n\nModel Loaded to device: {device}\n\n")

        for param in model.parameters(): # setting the parameters to be trainable
            param.requires_grad = True
        
        lossfn = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(lr = LEARNING_RATE, params = model.parameters())
        scheduler = ExponentialLR(optimizer, gamma=0.9, verbose = True)
        
        print(f"\n**Model Initialized**\n")
        
        print(f"***Training Started***\n\n")
        print(f"Max Sequence Length: {MAX_LENGTH}\nBatch Size: {batch_size}\n")
        
        for epoch in range(EPOCHS):
            
            train_loss = 0
            acc = 0
            
            model.train()
            for batch, (input_ids, attention_mask, label) in enumerate(tqdm(train_dataloader)):
                pred = model(input_ids.squeeze().to(device), attention_mask.squeeze().to(device))
                
                loss = lossfn(pred, label.to(device))

                train_loss+=loss
                
                acc+=accuracy_fn(torch.argmax(label, dim = 1).to(device), pred.argmax(dim = 1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            

            train_loss/=len(train_dataloader) # over all the samples
            acc/=len(train_dataloader)
            
            model.eval()
            
            test_loss = 0
            test_acc = 0
            with torch.inference_mode():
                for batch, (input_ids_test, attention_mask_test, label_test) in enumerate(tqdm(test_dataloader)):
                    
                    ypred = model(input_ids_test.squeeze().to(device), attention_mask_test.squeeze().to(device))

                    # print("Test : ", y.shape)
                    test_loss+=lossfn(ypred, label_test.to(device))
                    test_acc+=accuracy_fn(torch.argmax(label_test, dim = 1).to(device), ypred.argmax(dim = 1))
            
                test_loss/=len(test_dataloader)
                test_acc/=len(test_dataloader)
            
            scheduler.step()

            if acc > train_acc:
                train_acc = acc
        
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model, f"{SAVE_PATH}/{name}_{fold}_epoch_{epoch+1}_train_acc_{train_acc:.3f}_test_acc_{best_acc:.3f}.pth")

            print(f"\nEnd of epoch: {epoch+1}\n \t {fold}\t train loss: {train_loss:.3f}\t train_acc: {acc:.3f}\t test loss: {test_loss:.3f}\t test acc: {test_acc:.3f}\n")
            with open(f"log.txt", 'a') as f:
                f.writelines(f"\nEnd of epoch: {epoch+1}\n \t train loss: {train_loss:.3f}\t train_acc: {acc:.3f}\t test loss: {test_loss:.3f}\t test acc: {test_acc:.3f}\n")
            f.close()
        average_acc+=best_acc
    average_acc/=5
    print(f"\nAverage Accuracy: {average_acc:.3f}\n")
    