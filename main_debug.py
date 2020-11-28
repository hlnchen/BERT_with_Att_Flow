# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# In this notebook we train our model: a BERT-like model with attention flow inspired by BiDAF.
# %% [markdown]
# Dependencies:

# %%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections, time, spacy, copy
from layers.bert_plus_bidaf import BERT_plus_BiDAF
from utils import data_processing
from torch.utils.data import DataLoader

# %% [markdown]
# This part should be data loading and processing.
# 
# Input: SQuAD dataset handler/url/json
# 
# Output: processed dict/list/whatever: train_question, train_context, train_answer

# %%
# train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
# val_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
# train_encodings =  data_processing.data_processing(train_url)
# val_encodings = data_processing.data_processing(val_url)

# %% [markdown]
# Create a smaller dataset for debugging

# # %%
# for key in train_encodings.keys():
#     train_encodings[key] = train_encodings[key][0:100]


# # %%
# for key in val_encodings.keys():
#     val_encodings[key] = val_encodings[key][0:100]

# %% [markdown]
# Templates for S/L to save time preprocessing

# # %%
# torch.save(train_encodings,r'D:\OneDrive\Courses\ECS289 NLP\train_encodings.pt')
# torch.save(val_encodings,r'D:\OneDrive\Courses\ECS289 NLP\val_encodings.pt')


# %%
train_encodings = torch.load(r'D:\OneDrive\Courses\ECS289 NLP\train_encodings.pt')
val_encodings = torch.load(r'D:\OneDrive\Courses\ECS289 NLP\val_encodings.pt')


# %%
class SquadDataset(torch.utils.data.Dataset):
  def __init__(self,encodings):
    self.encodings = encodings
  def __getitem__(self,idx):
    return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)


# %%
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

# %% [markdown]
# This part should be model construction.

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# %%
model = BERT_plus_BiDAF(if_extra_modeling=True)


# %%
model.to(device)

# %% [markdown]
# This part should be declaration of the optimizer and the loss function. 

# %%
parameters = model.parameters()
print("Parameters to learn:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print("\t", name)
optimizer = optim.Adam(parameters, lr=5e-5)

# %% [markdown]
# This part should be the definition of training process:

# %%
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
    start = torch.argmax(max_in_row, dim=-1)
    end = torch.argmax(max_in_col, dim=-1)
    # check if indices are greater than no answer probability by threshold
    p_na = p_joint[:,0,0]
    max_prob = torch.max(max_row,dim=-1)
    start[p_na + threshold > max_prob] = 0
    end[p_na + threshold > max_prob] = 0
    # adjust to the encoding structure
    start[start!=0] += 63
    end[end!=0] += 63
    return start, end


# %%
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
            running_loss += loss

        epoch_loss = running_loss
        print('Loss: {:.4f}'.format(epoch_loss))

    # Output info after training
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return copy.deepcopy(model.state_dict())


# %%
nlp = spacy.blank("en")
def word_tokenize(sent, nlp):
    doc = nlp(sent)
    return [token.text for token in doc]


# %%
def compute_f1(a_gold, a_pred):
    gold_toks = word_tokenize(a_gold)
    pred_toks = word_tokenize(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# %%
def evaluate(model, eval_dataset, answers, threshold=0.1):
    """ TODO: debug"""
    n = len(eval_dataset)
    exact_match = 0
    f1_sum = 0
    model.eval()
    for i in range(n):
        input_ids = eval_dataset[i]['input_ids']
        attention_mask = eval_dataset[i]['attention_mask']
        golden_answer = answers[i]['text']

        _, start_logits, end_logits = model(torch.unsqueeze(input_ids,0), torch.unsqueeze(attention_mask,0))

        # compute null score and make prediction:
        start, end = predict_index(start_logits, end_logits, threshold)
        if start == 0 and end == 0:
            prediction = ""
        else:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            prediction = ' '.join(tokens[start:end+1])
        
        #exact match
        if(prediction == golden_answer):
            exact_match = exact_match + 1
        #F1_score
        f1_sum = f1_sum + get_F1_score(golden_answer, prediction)       
    accuracy = exact_match/n
    f1 = f1_sum / n
    return accuracy, f1

# %% [markdown]
# Rest part is for experiments:

# %%
dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)


# %%
trained_model = train(model, optimizer, dataloader, num_epochs=3)


# %%
em, f1 = evaluate(trained_model, val_dataset, )


# %%



