""" 
In this file we process the SQuAD dataset:
1. download
2. tokenize
3. add padding to both context and question

Input: SQuAD handler/url/json.
Output: encodings. encodings is a dictionary with keys including 'input_ids','attention_mask' and 'token_type'
and input_ids is [CLS][question][SEP][context][SEP]

if question has no answer, text of answer is 'no-answer'. start_positions and end_positions is [CLS] position, which is 0
"""

"""
This preprocess add padding to both context and question, making each context and each question has the same length to fit the BiDAF model
This may cause some influence to BERT model performance. 
Firstly, we have truncate more tokens of input_ids (up to 60 tokens in this training case) to fit in BERT max input length (512). It may lose some information.
Secondly, it may affect positional embedding

We may figure out better ways to avoid information lose when doing truncation

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
QUESTION_MAXLENGTH_SETTING = 62    # we can adjust this setting； If question over that length, do truncation
MAX_LENGTH = 512  #max input length that bert model can accept

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
def getQuestionLength(encodings, i):
    ans = 0
    for id in encodings['input_ids'][i]:
        if id == SEP_TOKEN:
            break
        else:
            ans += 1
    return ans

#add padding to make all questions have same length
def addPaddingQuestion(encodings,index,maxQueLen):
    queLen = getQuestionLength(encodings,index)
    for insertIndex in range(queLen,maxQueLen):
        encodings['input_ids'][index].insert(insertIndex,0)
        encodings['attention_mask'][index].insert(insertIndex,0)
    
    return maxQueLen - queLen


"""
add padding to both context and question
"""
def postTokenize(encodings):
    paddingLengths = []  #token_position of answer will be move to right since we add padding to question,
                        #paddingLength is to track how many positions moved

    #get max question length
    n = len(encodings['input_ids'])

    """
    maxQueLen = 0
    for index in range(n):
        queLen = getQuestionLength(encodings,index)
        if queLen > maxQueLen:
            maxQueLen = queLen
    """

    """
    for index in range(n):
        paddingLength = addPaddingQuestion(encodings,index,maxQueLen)
        paddingLengths.append(paddingLength)
    """

    for index in range(n):
        que_length = getQuestionLength(encodings,index)
        if que_length > QUESTION_MAXLENGTH_SETTING:
            encodings['input_ids'][index] = encodings['input_ids'][index][0:QUESTION_MAXLENGTH_SETTING] + encodings['input_ids'][index][que_length:]
            encodings['attention_mask'][index] = encodings['attention_mask'][index][0:QUESTION_MAXLENGTH_SETTING] + encodings['attention_mask'][index][que_length:]
        else:
            for insertIndex in range(que_length,QUESTION_MAXLENGTH_SETTING):
                encodings['input_ids'][index].insert(insertIndex,0)
                encodings['attention_mask'][index].insert(insertIndex,0)

        paddingLength = QUESTION_MAXLENGTH_SETTING - que_length
        paddingLengths.append(paddingLength)
    

    """
    if input_length > 512, do truncation to 512
    if input_length < 512, add padding to 512
    """    

    for index in range(n):
        lst_length = len(encodings['input_ids'][index])
        if lst_length > 512:
            encodings['input_ids'][index] = encodings['input_ids'][index][0:512]
            encodings['attention_mask'][index] = encodings['attention_mask'][index][0:512]
        else:
            for insertIndex in range(lst_length,512):
                encodings['input_ids'][index].insert(insertIndex,0)
                encodings['attention_mask'][index].insert(insertIndex,0)

    return paddingLengths


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
    for i in range(len(answers)):
        if answers[i]['text'] == 'no-answer':
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(encodings.char_to_token(i,answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i,answers[i]['answer_end']-1))
            #if none, the answer span has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions':start_positions,'end_positions':end_positions})

"""
after padding to QUESTION, token_position of answer will move to right
so we need modify token_positions of answer
"""

def modify_token_positions(encodings, paddingLengths, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        if answers[i]['text'] == 'no-answer':
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_position = encodings['start_positions'][i] + paddingLengths[i]
            end_position = encodings['end_positions'][i] + paddingLengths[i]
            if start_position > 511:
                start_position = 511
            if end_position > 511:
                end_position = 511
            start_positions.append(start_position)
            end_positions.append(end_position)
    
    encodings.update({'start_positions':start_positions,'end_positions':end_positions})    


def data_processing(url):
    response = urllib.request.urlopen(url)
    raw = pd.read_json(response)
    contexts, questions, answers, ids = load_data(raw)
    add_end_idx(answers,contexts)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    encodings = tokenizer(questions, contexts, truncation = True, padding = True)
    add_token_positions(encodings,answers,tokenizer)
    paddingLengths = postTokenize(encodings)
    modify_token_positions(encodings,paddingLengths,answers)

    return encodings, answers

if __name__ == "__main__":
    #union test and utilize example below
    encodings =  data_processing("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")

    print("length of start_postion:",len(encodings['start_positions']))
    print("start_position:",encodings['start_positions'][0])

    print("length of input_ids",len(encodings['input_ids'][0]))
    print("length of mask",len(encodings['attention_mask'][0]))


