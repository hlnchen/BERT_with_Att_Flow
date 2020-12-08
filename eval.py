import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections, time, spacy, copy
from layers.bert_plus_bidaf import BERT_plus_BiDAF
from utils import data_processing
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
#import nltk
#nltk.download('punkt')
# In[16]:
val_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
val_encodings, val_answer = data_processing.data_processing(val_url)
torch.save(val_answer,r'val_answer.pt')
# In[]
# val_encodings = torch.load(r'D:\OneDrive\Courses\ECS289 NLP\val_encodings.pt')
# val_answer=torch.load(r'val_answer.pt')
# In[17]:
class SquadDataset(torch.utils.data.Dataset):
  def __init__(self,encodings):
    self.encodings = encodings
  def __getitem__(self,idx):
    return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)


# In[18]:
val_dataset = SquadDataset(val_encodings)
# This part should be model construction.

# In[19]:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# In[20]:
model = BERT_plus_BiDAF(if_extra_modeling=True)
model.load_state_dict(torch.load('BERT_model'))
model.to(device)
print("Model imported successfully")
# In[21]:
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
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
    max_prob,_ = torch.max(max_row,dim=-1)
    start[p_na + threshold > max_prob] = 0
    end[p_na + threshold > max_prob] = 0
    return start, end

# In[24]:
nlp = spacy.blank("en")
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]
# In[26]:
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
    recall = 1.0 * num_same / len(gold_toks)s
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
# In[27]:
def evaluate(model, eval_dataset, answers, threshold=0.1):
    """ TODO: debug"""
    n = len(eval_dataset)
    exact_match = 0
    f1_sum = 0
    model.eval()
    for i in range(n):
        if i%1000==0:
            print('evaluated {}/{}:'.format(i, n))
        input_ids = eval_dataset[i]['input_ids']
        attention_mask = eval_dataset[i]['attention_mask']
        golden_answer = answers[i]['text']
        ipid = torch.unsqueeze(input_ids,0)
        attm =torch.unsqueeze(attention_mask,0)
        with torch.cuda.device(0):
            ipid = ipid.cuda(async=True)  # in test loader, pin_memory = True
            attm = attm.cuda(async=True)
        _, start_logits, end_logits = model(ipid, attm)

        # compute null score and make prediction:
        start, end = predict(torch.unsqueeze(start_logits,dim=0),torch.unsqueeze(end_logits,dim=0), threshold)
        # adjust to our context paddings
        start[start!=0] += 62
        end[end!=0] += 62
        if start == 0 and end == 0:
            prediction = ""
        else:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            prediction = ' '.join(tokens[start:end + 1])

            # exact match
        if (prediction == golden_answer):
            exact_match = exact_match + 1
            # F1_score
        f1_sum = f1_sum + compute_f1(golden_answer, prediction)
    accuracy = exact_match / n
    f1 = f1_sum / n
    return accuracy, f1
# In[]:
em, f1 = evaluate(model, val_dataset, val_answer)

print("accuracy: ")
print(em)
print("f1 score: ")
print(f1)