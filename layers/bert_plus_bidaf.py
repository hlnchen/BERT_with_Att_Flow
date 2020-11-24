""" 
This file contains the definition of our main model class:

Input: the input of BERT

Output: [answer_start], [answer_end]
tensor of size [batch_size, context_length, 1]
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.att_flow import AttFlow
from layers.char_cnn import CharCNN
from layers.pred_layer import PredictionLayer
from pytorch_pretrained_bert import BertModel
# from transformers import BertPreTrainedModel, BertModel, BertTokenizer

class BERT_plus_BiDAF(nn.Module):
    def __init__(self, if_cnn = False, if_extra_modeling = False):
        super().__init__()
        # Constants
        self.hidden_dim = 768   # dimension d: because of BERT
        self.vocab_size = None  # TODO: use BertTokenizer.vocab_size
        self.question_len = 62
        # Network modules
        # BERT
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        # CNN
        """ TODO: add CNN embedding layer"""
        if if_cnn:
            self.cnn = CharCNN(input_length=512, vocab_size=self.vocab_size) # because of BERT
        else:
            self.cnn = None

        # Bidirectional attention
        if self.cnn:
            self.attention_layer = AttFlow(feature_dimension=2*self.hidden_dim)
        else:
            self.attention_layer = AttFlow(feature_dimension=self.hidden_dim)
        
        # Additional modeling layer LSTM/Transformer:
        """ TODO: add a flag using BERT or LSTM"""
        if if_extra_modeling:
            if self.cnn:
                self.modeling_layer = nn.LSTM(input_size=2*self.hidden_dim,hidden_size=2*self.hidden_dim)
            else:
                self.modeling_layer = nn.LSTM(input_size=self.hidden_dim,hidden_size=self.hidden_dim)
            

        # Prediction
        if self.cnn:
            self.prediction_layer = PredictionLayer(feature_dimension=8*self.hidden_dim)
        else: 
            self.prediction_layer = PredictionLayer(feature_dimension=4*self.hidden_dim) 

    def forward(self, input_ids, input_mask):
        """ 
        Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length](N,T) with the word token indices in the vocabulary
        `input_mask`: an optional torch.LongTensor of shape (N,T) with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        """
        # Feed into BERT
        bert_features, _ = self.bert_layer(input_ids = input_ids, token_type_ids = None, attention_mask = input_mask, output_all_encoded_layers=False) # (N,L,d)
        # Separate features
        bert_question_features = bert_features[:, 1:self.question_len,:] # (N,J,d)
        bert_context_features = torch.cat((bert_features[:, 0, :], bert_features[:, self.question_len:, :]), dim = 1) # (N,T,d), NOTE: T+J = 512
        
        
        # Feed into CNN
        if self.cnn:
            """ TODO: check the validity here """
            cnn_features = self.cnn(input_ids)
            """ TODO: separate CNN features """
            cnn_context_features = None
            cnn_question_features = None

        # Concatenate and feed into attention
        if self.cnn:
            context_features = torch.cat((bert_context_features, cnn_context_features), dim = -1) # (N,T,2d)
            question_features = torch.cat((bert_question_features, cnn_question_features), dim = -1) # (N,T,2d)
            c2q_attention, q2c_attention = self.attention_layer(context_features, question_features) # (N,T,2d), (N,T,2d)
        else:
            c2q_attention, q2c_attention = self.attention_layer(bert_context_features, bert_question_features) # (N,T,d), (N,T,d)

        # If we use extra modeling layer
        if self.modeling_layer:
            """ TODO: add modeling layer"""
            None
        
        # Combine all features and make prediction
        if self.cnn:
            combined_features = torch.cat((context_features, c2q_attention, 
            torch.mul(context_features, c2q_attention), torch.mul(context_features, q2c_attention)), dim=2)
        else:
            combined_features = torch.cat((bert_context_features, c2q_attention, 
            torch.mul(bert_context_features, c2q_attention), torch.mul(bert_context_features, q2c_attention)), dim = -1) # (N,T,4d)
        p_start, p_end = self.prediction_layer(combined_features)

        return p_start, p_end

if __name__ == "__main__":
    model = BERT_plus_BiDAF()
    print(model)