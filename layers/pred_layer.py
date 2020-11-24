"""
This file contains the definition of the prediction layer.
The layer should take in all features, then output the result of our model.
If answer_start = answer_end = 0, then predict no answer.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PredictionLayer(nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        self.pred_start = nn.Linear(in_features=feature_dimension, out_features=1, bias= False)
        self.pred_end = nn.Linear(in_features=feature_dimension, out_features=1, bias = False)
    def forward(self, features):
        """ 
        Input:
        Final features of shape [batch_size, sequence_len, feature_dim](N,T,d)
        Output:
        p_start, p_end of shape [batch_size, sequence_len, 1] which indicates 
        the probability that the word at each position is the start/end of the answer span.
        """
        p_start = F.softmax(self.pred_start(features).squeeze(), dim = -1) # softmax((N,T))
        p_end = F.softmax(self.pred_end(features).squeeze(), dim = -1) # softmax((N,T))
        return p_start, p_end