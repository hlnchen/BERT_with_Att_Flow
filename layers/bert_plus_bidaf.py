""" 
This file contains the definition of our main model class:

Input: 
the batch of [question, context];
tensor of size [batch_size, token_dimension, len_question + len_context].

Output: [answerability_score, answer_start, answer_end]
tensor of size [batch_size, 3]

NOTE: answerability_score may not be necessary and 
needs to be reconsidered based on dicussion and implementation.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from att_flow import AttFlow
from char_cnn import CharCNN

"""
TODO: check if there is a pretrained BERT as a torch.nn.module for customized models:
https://github.com/maknotavailable/pytorch-pretrained-BERT 
seems to be a good choice

Input of BERT: 
the batch of [question, context] (with BERT special tokens added);
tensor of size [batch_size, token_dimension, len_question + len_context + len_tokens]

Output of BERT:
the batch of [question_BERT_features, context_BERT_features]
tensor of size [batch_size, feature_dimension, len_question + len_context]
(if special token not used)
"""

class BERT_plus_BiDAF(nn.Module):
    def __init__(self, input):
        """ TODO: 
        determine input
        determine members
        """
    def forward(self, input):
        """ TODO: 
        determine input 
        construct the computation graph
        """