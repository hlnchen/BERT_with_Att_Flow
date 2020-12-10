import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections, time, spacy, copy, string, re, json
from layers.bert_plus_bidaf import BERT_plus_BiDAF
from utils import data_processing
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
# %%
# val_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
# val_encodings, val_answer = data_processing.data_processing(val_url)
# torch.save(val_answer,r'D:\OneDrive\Courses\ECS289 NLP\val_answer.pt')
# torch.save(val_encodings,r'D:\OneDrive\Courses\ECS289 NLP\val_encodings.pt')

# %%
class SquadDataset(torch.utils.data.Dataset):
  def __init__(self,encodings):
    self.encodings = encodings
  def __getitem__(self,idx):
    return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)

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

def word_tokenize(sent, nlp):
    doc = nlp(sent)
    return [token.text for token in doc]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred, nlp):
    gold_toks = word_tokenize(a_gold, nlp)
    pred_toks = word_tokenize(a_pred, nlp)
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
# def evaluate(model, eval_dataset, answers, threshold=0.1):
#     n = len(eval_dataset)
#     exact_match = 0
#     f1_sum = 0
#     model.eval()
#     with torch.no_grad():
#         for i in range(n):
#             if i%1000==0:
#                 print('evaluated {}/{}:'.format(i, n))
#             input_ids = eval_dataset[i]['input_ids']
#             attention_mask = eval_dataset[i]['attention_mask']
#             ipid = torch.unsqueeze(input_ids,0)
#             attm = torch.unsqueeze(attention_mask,0)
            
#             # golden_answer = normalize_answer(answers[i]['text']) 
#             tokens = tokenizer.convert_ids_to_tokens(input_ids)
#             golden_start, golden_end = eval_dataset[i]['start_positions'], eval_dataset[i]['end_positions']
#             if golden_start == 0 and golden_end == 0:
#                 golden_answer = "noanswer"
#             else:
#                 golden_answer = normalize_answer(' '.join(tokens[golden_start:golden_end + 1]))


#             with torch.cuda.device(0):
#                 ipid = ipid.cuda(non_blocking=True)  # in test loader, pin_memory = True
#                 attm = attm.cuda(non_blocking=True)
#             _, start_logits, end_logits = model(ipid, attm)

#             # compute null score and make prediction:
#             start, end = predict(torch.unsqueeze(start_logits,dim=0),torch.unsqueeze(end_logits,dim=0), threshold)
#             # adjust to the context padding
#             start[start!=0] += 62
#             end[end!=0] += 62
#             # print("sample ",i,': ', start, end)
#             # print("true ", i, ': ', eval_dataset[i]['start_positions'], eval_dataset[i]['end_positions'])
#             if start == 0 and end == 0:
#                 prediction = "noanswer"
#             else:
#                 prediction = normalize_answer(' '.join(tokens[start:end + 1]))

#             # exact match
#             if (prediction == golden_answer):
#                 exact_match = exact_match + 1
#             # F1_score
#             f1_sum = f1_sum + compute_f1(golden_answer, prediction)
#     accuracy = exact_match / n
#     f1 = f1_sum / n
#     return accuracy, f1
def evaluate(model, eval_dataset):
    """ To obtain the predictions, run this function"""
    n = len(eval_dataset)
    exact_match = 0
    f1_sum = 0
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(n):
            if i%1000==0:
                print('evaluated {}/{}:'.format(i, n))
            input_ids = eval_dataset[i]['input_ids']
            attention_mask = eval_dataset[i]['attention_mask']
            ipid = torch.unsqueeze(input_ids,0)
            attm = torch.unsqueeze(attention_mask,0)
            with torch.cuda.device(0):
                ipid = ipid.cuda(non_blocking=True)  # in test loader, pin_memory = True
                attm = attm.cuda(non_blocking=True)
            _, start_logits, end_logits = model(ipid, attm)

            logit = {'start_logits': start_logits, 'end_logits': end_logits}
            logits.append(logit)
    # with open('prediction_thresholds_'+str(threshold)+'.json', 'w') as fout:
    #     json.dump(predictions, fout)
    torch.save(logits, 'pred_logits.pt')
    return logits

def compare(logits, eval_dataset, tokenizer, nlp, threshold):
    """ To compare the predictions and answers, run this function"""
    n = len(eval_dataset)
    exact_match = 0
    f1_sum = 0
    for i in range(n):
        if i%1000==0:
            print('compared {}/{}:'.format(i, n))

        input_ids = eval_dataset[i]['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        golden_start, golden_end = eval_dataset[i]['start_positions'], eval_dataset[i]['end_positions']
        if golden_start == 0 and golden_end == 0:
            golden_answer = "noanswer"
        else:
            golden_answer = normalize_answer(' '.join(tokens[golden_start:golden_end + 1]))
        
        start_logits, end_logits = logits[i]['start_logits'], logits[i]['end_logits']
        # compute null score and make prediction:
        pred_start, pred_end = predict(torch.unsqueeze(start_logits,dim=0),torch.unsqueeze(end_logits,dim=0), threshold)
        # adjust to the context padding
        pred_start[pred_start!=0] += 62
        pred_end[pred_end!=0] += 62
        if pred_start == 0 and pred_end == 0:
                pred_answer = "noanswer"
        else:
            pred_answer = normalize_answer(' '.join(tokens[pred_start:pred_end + 1]))
        if pred_answer == golden_answer:
            exact_match += 1
        f1_sum += compute_f1(golden_answer, pred_answer, nlp)

    acc = 100 * exact_match / n
    f1_score = 100 * f1_sum /n
    return acc, f1_score
# %%

if __name__ == "__main__":
    val_encodings = torch.load(r'D:\OneDrive\Courses\ECS289 NLP\val_encodings.pt')
    val_answer=torch.load(r'D:\OneDrive\Courses\ECS289 NLP\val_answer.pt')
    val_dataset = SquadDataset(val_encodings)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # model = BERT_plus_BiDAF(if_extra_modeling=True)
    # model.load_state_dict(torch.load(r'D:\OneDrive\Courses\ECS289 NLP\bert_BiDAF.pt'))
    # model = model.to(device)
    # print("Model imported successfully")
    
    nlp = spacy.blank("en")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    predictions = torch.load('pred_logits.pt')
    # predictions = evaluate(model, val_dataset)

    threshold = np.arange(0,0.11,0.01)
    accs, f1s = [], []
    for i in range(len(threshold)):
        print("Compare with threshold = ", str(threshold[i]))
        acc, f1 = compare(predictions, val_dataset, tokenizer, nlp, threshold[i])
        accs.append(acc)
        f1s.append(f1)
        print("accuracy: ")
        print(acc)
        print("f1 score: ")
        print(f1)

    print("All accuracy: ")
    print(accs)
    print("All F1 scores: ")
    print(f1s)

    # em, f1 = evaluate(model, val_dataset, val_answer, threshold=0.01)
    # print("EM = {}, F1 = {}".format(em,f1))
