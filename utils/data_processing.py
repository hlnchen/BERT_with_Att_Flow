""" 
In this file we process the SQuAD dataset:
1. download
2. tokenize
3. add padding to both context and question

Input: SQuAD handler/url/json.
Output: encodings. encodings is a dictionary with keys including 'input_ids','attention_mask' and 'token_type'
and input_ids is [CLS][context][SEP][question][SEP]

if question has no answer, text of answer is 'no-answer'. start_positions and end_positions is [CLS] position, which is 0
"""

"""
This preprocess add padding to both context and question, making each context and each question has the same length to fit the BiDAF model
This may cause some influence to BERT model performance. One of the reason is it may affect positional embedding
"""

import numpy as np
import pandas as pd
import json

import requests
import urllib

from transformers import BertTokenizer, BertForQuestionAnswering, BertTokenizerFast
import torch

CLS_TOKEN = 101
SEP_TOKEN = 102

def load_data(train_df):
    contexts = []
    questions = []
    answers = []
    ids = []
    for i in range(train_df['data'].shape[0]):
        topic = train_df['data'].iloc[i]['paragraphs']
        for sub_para in topic:
            context = sub_para['context']
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                contexts.append(context)
                ids.append(q_a['id'])
                if q_a['is_impossible'] is False:
                    answers.append(q_a['answers'][0])
                else:
                    answer = {}
                    answer['answer_start'] = 0
                    answer['text'] = 'no-answer'
                    answers.append(answer)

    return contexts, questions, answers, ids

"""
there is only start index in SQuAD dataset. add end_index
"""
def add_end_idx(answers, contexts):
    for answer, context in zip(answers,contexts):
        if answer['text'] == "no-answer":
            answer['answer_end'] = 0
        else:
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)
            #sometimes SQuAD answers are off by a character or two 
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1
            elif context[start_idx-2: end_idx-2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2

"""
below functions are to padding both contexts and question to make each contexts and each questions have the same length
"""
def getContextLength(encodings, i):
    ans = 0
    for id in encodings['input_ids'][i]:
        if id == SEP_TOKEN:
            break
        else:
            ans += 1
    return ans

#add padding to make all questions have same length
def addPaddingContext(encodings,index,maxConLen):
    conLen = getContextLength(encodings,index)
    for insertIndex in range(conLen,maxConLen):
        encodings['input_ids'][index].insert(insertIndex,0)
        encodings['attention_mask'][index].insert(insertIndex,0)

def getQuestionLength(encodings,index):
    SEP_SIGNAL = 0
    ans = 0
    for id in encodings['input_ids'][index]:
        if SEP_SIGNAL == 0 and id == SEP_TOKEN:
            SEP_SIGNAL = 1
        if SEP_SIGNAL == 1:
            ans += 1
    return ans

def addPaddingQuestion(encodings,index,maxQueLen):
    queLen = getQuestionLength(encodings,index)
    for insertIndex in range(queLen,maxQueLen):
        encodings['input_ids'][index].insert(insertIndex,0)
        encodings['attention_mask'][index].insert(insertIndex,0)

"""
add padding to both context and question
"""
def postTokenize(encodings):
    #get max context length
    n = len(encodings['input_ids'])
    maxConLen = 0
    for index in range(n):
        conLen = getContextLength(encodings,index)
        if conLen > maxConLen:
            maxConLen = conLen

    #now we have maxConLen
    for index in range(n):
        addPaddingContext(encodings,index,maxConLen)

    #get max question length
    maxQueLen = 0
    for index in range(n):
        queLen = getQuestionLength(encodings,index)
        if queLen > maxQueLen:
            maxQueLen = queLen
    #now we have maxQueLen
    for index in range(n):
        addPaddingQuestion(encodings,index,maxQueLen)
"""
end of padding functions 
"""

"""
start_position in SQuAD is character_position
transfer to token_position
"""
def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    #TO DO
    #consider no answer situation
    if answers['text'] == 'no-answer':
        start_positions.append(0)
        end_positions.append(0)
    else:
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i,answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i,answers[i]['answer_end']-1))
            #if none, the answer span has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions':start_positions,'end_positions':end_positions})

def data_processing(url):
    response = urllib.request.urlopen(url)
    raw = pd.read_json(response)
    contexts, questions, answers, ids = load_data(raw)
    add_end_idx(answers,contexts)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    encodings = tokenizer(contexts, questions, truncation = True, padding = True)
    add_token_positions(encodings,answers,tokenizer)
    postTokenize(encodings)

    return encodings

#union test and utilize example below
encodings =  data_processing("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")