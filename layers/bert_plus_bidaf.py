""" 
This file contains the definition of our main model class:

Input: the input of BERT

Output: 
p_start, p_end of shape [batch_size, sequence_len, 1] which indicates 
the probability that the word at each position is the start/end of the answer span.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_pretrained_bert import BertModel
try:
    from layers.att_flow import AttFlow
    from layers.char_cnn import CharCNN
    from layers.pred_layer import PredictionLayer
except ModuleNotFoundError:
    from att_flow import AttFlow
    from char_cnn import CharCNN
    from pred_layer import PredictionLayer
# from transformers import BertPreTrainedModel, BertModel, BertTokenizer

class BERT_plus_BiDAF(nn.Module):
    def __init__(self, question_len = 62, if_cnn = False, if_extra_modeling = False):
        """
        question_len: the length of the question in the padded sequence
        if_cnn: if to use a character-level cnn encoder. Default: false
        if_extra_modeling: if to use an additional LSTM modeling layer after attention flow. Default: false
        TODO: consider add a flag replacing the LSTM modeling by transformer
        """
        super().__init__()
        # Constants
        self.hidden_dim = 768   # dimension d: because of BERT
        self.vocab_size = None  # TODO: use BertTokenizer.vocab_size
        self.question_len = question_len
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
        if if_extra_modeling:
            if self.cnn:
                self.modeling_layer = nn.LSTM(input_size=8*self.hidden_dim,hidden_size=2*self.hidden_dim, num_layers=2)
            else:
                self.modeling_layer = nn.LSTM(input_size=4*self.hidden_dim,hidden_size=2*self.hidden_dim, num_layers=2)
        else:
            self.modeling_layer = None
            
        # Prediction
        if self.modeling_layer:
            self.prediction_layer = PredictionLayer(feature_dimension=2*self.hidden_dim)
        elif self.cnn:
            self.prediction_layer = PredictionLayer(feature_dimension=8*self.hidden_dim)
        else:
            self.prediction_layer = PredictionLayer(feature_dimension=4*self.hidden_dim)

    def forward(self, input_ids, input_mask, start_pos = None, end_pos = None):
        """ 
        Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length](N,T) with the word token indices in the vocabulary
        `input_mask`: an optional torch.LongTensor of shape (N,T) with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_pos`: the start of the answer span [batch_size]
        `end_pos`: the end of the answer span [batch_size]
        """
        # Feed into BERT
        bert_features, _ = self.bert_layer(input_ids = input_ids, token_type_ids = None, attention_mask = input_mask, output_all_encoded_layers=False) # (N,L,d)
        # Separate features
        bert_question_features = bert_features[:, 1:self.question_len+1,:] # (N,J,d) J=62
        bert_context_features = torch.cat((bert_features[:, 0, :].unsqueeze(dim=1), bert_features[:, self.question_len+1:, :]), dim=1) # (N,T,d), T=450 NOTE: T+J = 512， context is [CLS][SEP][context][SEP]
        
        
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
            c2q_attention, q2c_attention = self.attention_layer(bert_context_features, bert_question_features) # (N,T,d), (N,J,d)
        
        # Combine all features and make prediction
        if self.cnn:
            combined_features = torch.cat((context_features, c2q_attention, 
            torch.mul(context_features, c2q_attention), torch.mul(context_features, q2c_attention)), dim=-1) # (N,T,8d)
        else:
            combined_features = torch.cat((bert_context_features, c2q_attention, 
            torch.mul(bert_context_features, c2q_attention), torch.mul(bert_context_features, q2c_attention)), dim = -1) # (N,T,4d)

        # If we use extra modeling layer
        if self.modeling_layer:
            combined_features = self.modeling_layer(combined_features)[0] #(N,T,2d)
        
        start_logits, end_logits = self.prediction_layer(combined_features) # (N,T), (N,T)
        if len(start_logits.shape == 1):
            start_logits.unsqueeze(dim=0)
            end_logits.unsqueeze
        total_loss = None
        # Compute loss
        if start_pos is not None and end_pos is not None:
            # adjust to our context paddings:
            start_pos[start_pos!=0] -= self.question_len
            end_pos[end_pos!=0] -= self.question_len

            loss = nn.CrossEntropyLoss()
            start_loss = loss(start_logits, start_pos)
            end_loss = loss(end_logits, end_pos)
            total_loss = (start_loss + end_loss)/2
        
        return total_loss, start_logits, end_logits

if __name__ == "__main__":
    model = BERT_plus_BiDAF(if_extra_modeling=True)
    print(model)
    