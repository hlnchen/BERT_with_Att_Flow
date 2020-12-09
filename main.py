"""
.py version of main
since it's more convenient to run .py file on GCP
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections, time, spacy, copy
from layers.bert_plus_bidaf import BERT_plus_BiDAF
from utils import data_processing
from torch.utils.data import DataLoader

train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
train_encodings, _ =  data_processing.data_processing(train_url)
#create a smaller dataset for trying
for key in train_encodings.keys():
    train_encodings[key] = train_encodings[key][0:200]

class SquadDataset(torch.utils.data.Dataset):
  def __init__(self,encodings):
    self.encodings = encodings
  def __getitem__(self,idx):
    return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_plus_BiDAF(if_extra_modeling=True)
model.to(device)
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=5e-5)

#below is the definition of training process
def predict(logits_start, logits_end, threshold = 0.1):
    """
    Input:
    logits_start, logits_end: torch.tensor() of shape [batch_size, sequence length]
    return the index i,j such that i<=j and logits_start[i]+logits[j] is maximized
    """
    # compute probability
    p_start = F.softmax(logits_start, dim=-1)
    p_end = F.softmax(logits_end, dim=-1)
    # compute joint probability
    p_joint = torch.triu(torch.bmm(p_start.unsqueeze(dim=2), p_end.unsqueeze(dim=1)))
    # get the batchwise indices
    max_row, _ = torch.max(p_joint, dim=2)
    max_col, _ = torch.max(p_joint, dim=1)
    start = torch.argmax(max_row, dim=-1)
    end = torch.argmax(max_col, dim=-1)
    # check if indices are greater than no answer probability by threshold
    p_na = p_joint[:,0,0]
    max_prob = torch.max(max_row,dim=-1)
    start[p_na + threshold > max_prob] = 0
    end[p_na + threshold > max_prob] = 0
    # adjust to the encoding structure
    start[start!=0] += 62
    end[end!=0] += 62
    return start, end

def train(model, optimizer, dataloader, num_epochs = 3):
    """
    Inputs:
    model: a pytorch model
    dataloader: a pytorch dataloader
    loss_func: a pytorch criterion, e.g. torch.nn.CrossEntropyLoss()
    optimizer: an optimizer: e.g. torch.optim.SGD()
    """
    start = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1))
        print('-'*10)
        # Each epoch we make a training and a validation phase
        model.train()
            
        # Initialize the loss and binary classification error in each epoch
        running_loss = 0.0

        for batch in dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            # Send data to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            # Forward computation
            # Get the model outputs
            outputs = model(input_ids, attention_mask, start_positions, end_positions)
            loss = outputs[0]
            # In training phase, backprop and optimize
            loss.backward()
            optimizer.step()                   
            # Compute running loss/accuracy
            running_loss += loss.item() * input_ids.size(0)

        epoch_loss = running_loss
        print('Loss: {:.4f}'.format(epoch_loss))

    # Output info after training
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model.state_dict()


dataloader = DataLoader(train_dataset,batch_size=2,shuffle=True)

trained_model = train(model, optimizer, dataloader, num_epochs=30)
torch.save(trained_model,'bert_BiDAF.pt')