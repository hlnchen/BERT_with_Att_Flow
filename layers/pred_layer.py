"""
This file contains the definition of the prediction layer.
The layer should take in all features, then output the result of our model.

Input:
the batch of final features;
tensor of size [batch_size, len_seq, combined_feature_dimension]

Output:
Output: [answer_start, answer_end]
tensor of size [batch_size, 2]
If answer_start = answer_end = 0, then predict no answer.

NOTE: answerability_score may not be necessary and 
needs to be reconsidered based on dicussion and implementation.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PredictionLayer(nn.Module):
    def __init__(self, feature_dimension):
        self.pred_start = nn.Linear(in_features=feature_dimension, out_features=1, bias= False)
        self.pred_end = nn.Linear(in_features=feature_dimension, out_features=1, bias = False)
    def forward(self, features):
        p_start = F.softmax(self.pred_start(features).squeeze(), dim = -1)
        p_end = F.softmax(self.pred_end(features).squeeze(), dim = -1)
        return p_start, p_end