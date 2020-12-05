import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections, time, sys
from layers.bert_plus_bidaf import BERT_plus_BiDAF
from utils import data_processing
from torch.utils.data import DataLoader

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self,encodings):
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def train(device, model, optimizer, dataloader, num_epochs = 3):
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
            running_loss += loss

        epoch_loss = running_loss
        print('Loss: {:.4f}'.format(epoch_loss))

    # Output info after training
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model.state_dict()

def main(learing_rate = 5e-5, batch_size = 4, num_epochs = 3):
    train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    train_encodings, _ =  data_processing.data_processing(train_url)
    train_dataset = SquadDataset(train_encodings)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = BERT_plus_BiDAF(if_extra_modeling=True)
    model.to(device)
    print("Model Structure:","\n","-"*10)
    print(model)

    parameters = model.parameters()
    print("Parameters to learn:","\n","-"*10)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)
    
    print("Hyperparameters:","\n","-"*10)
    print("Learning Rate: ", learing_rate)
    print("Batch Size: ", batch_size)
    print("Number of Epochs: ", num_epochs)

    optimizer = optim.Adam(parameters, lr=learing_rate)
    dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    trained_model = train(device, model, optimizer, dataloader, num_epochs=num_epochs)
    torch.save(trained_model,'trained_model.pt')

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[0], sys.argv[1], sys.argv[2])
    else:
        main()