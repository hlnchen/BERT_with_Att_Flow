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
        logits_start, logits_end of shape [batch_size, sequence_len] which are used for compute
        the probability that the word at each position is the start/end of the answer span.
        """
        logits_start = self.pred_start(features).squeeze()# (N,T)
        logits_end = self.pred_end(features).squeeze() # (N,T)

        return logits_start, logits_end