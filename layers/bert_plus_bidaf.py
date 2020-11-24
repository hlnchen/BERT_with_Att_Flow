""" 
This file contains the definition of our main model class:

Input: 
the batch of [question, context];
tensor of size [batch_size, token_dimension, len_question + len_context].

Output: [answer_start, answer_end]
tensor of size [batch_size, 2]

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
from pred_layer import PredictionLayer
from pytorch_pretrained_bert import BertModel """ NOTE: transformers library seems to work"""
# from transformers import BertPreTrainedModel, BertModel, BertTokenizer
"""
Output of BERT: tuple of (encoded_layers, pooled_output)
`encoded_layers`: controled by `output_all_encoded_layers` argument:
- `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
    of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
    encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
- `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
    to the last attention block of shape [batch_size, sequence_length, hidden_size],

`pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
classifier pretrained on top of the hidden state associated to the first character of the
input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
"""

class BERT_plus_BiDAF(nn.Module):
    def __init__(self, if_cnn = False, if_extra_modeling = False):
        self.hidden_dim = 768   # because of BERT
        self.vocab_size = None  # TODO: use BertTokenizer.vocab_size
        # BERT
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        # CNN
        """ TODO: add CNN embedding layer"""
        if if_cnn:
            self.cnn = CharCNN(input_length=512, vocab_size=self.vocab_size) # because of BERT

        # Bidirectional attention
        if if_cnn:
            self.attention_layer = AttFlow(feature_dimension=2*self.hidden_dim)
        else:
            self.attention_layer = AttFlow(feature_dimension=self.hidden_dim)
        
        # Additional modeling layer LSTM/Transformer:
        """ TODO: add additional modeling layer(not urgent)"""
        if if_extra_modeling:
            self.modeling_layer = None

        # Prediction
        """ TODO: check final dimension"""
        self.prediction_layer = PredictionLayer(feature_dimension=self.hidden_dim) 

    def forward(self, input_ids, input_mask):
        """ 
        Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
        `input_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        """
        # Feed into BERT
        bert_features, _ = self.bert_layer(input_ids = input_ids, token_type_ids = None, attention_mask = input_mask, output_all_encoded_layers=False)
        # Separate features
        """ TODO: separate BERT features"""
        bert_context_features = None
        bert_question_features = None
        
        # Feed into CNN
        if self.cnn:
            """ TODO: check the validity here """
            cnn_features = self.cnn(input_ids)
            """ TODO: separate CNN features """
            cnn_context_features = None
            cnn_question_features = None

        # Concatenate and feed into attention
        if self.cnn:
            """ TODO: concatenate two features """
            context_features = torch.cat((bert_context_features, cnn_context_features), dim = -1)
            question_features = torch.cat((bert_question_features, cnn_question_features), dim = -1)
            c2q_attention, q2c_attention = self.attention_layer(context_features, question_features)
        else:
            c2q_attention, q2c_attention = self.attention_layer(bert_context_features, bert_question_features)
        
        # Combine all features and make prediction
        """ TODO: build combined features"""
        combined_features = None
        p_start, p_end = self.prediction_layer(combined_features)

        return p_start, p_end