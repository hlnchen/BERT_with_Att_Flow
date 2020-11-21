""" 
In this file we process the SQuAD dataset:
1. download
2. tokenize
3. keep necessary information
4. anything else

Input: SQuAD handler/url/json.
Output: [question, context, answer] ready to fit in the model.
if question has no answer, answer in output will be "no-answer"
"""

import numpy as np
import pandas as pd
import json

import requests
import urllib

def load_data(train_df):
    contexts = []
    questions = []
    answers = []
    for i in range(train_df['data'].shape[0]):
        topic = train_df['data'].iloc[i]['paragraphs']
        for sub_para in topic:
            context = sub_para['context']
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                contexts.append(context)
                if q_a['is_impossible'] is False:
                    answers.append(q_a['answers'][0])
                else:
                    answers.append("no-answer")

                """
                for answer in q_a['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                """

    return contexts, questions, answers

def data_processing(url):
    response = urllib.request.urlopen(url)
    raw = pd.read_json(response)
    contexts, questions, answers = load_data(raw)

    return contexts, questions, answers

#union test and utilize example below
contexts,questions,answers =  data_processing("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
print(len(contexts))
print(len(questions))
print(len(answers))