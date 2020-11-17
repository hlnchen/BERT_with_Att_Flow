"""
This file contains the definition of the prediction layer.
The layer should take in all features, and do some operations
(needs to be determined, e.g. concatenation, product, softmax) according to
the prediction scheme we choose, then output the result of our model.

Input:
the batch of [question_features, context_features, attention_features];
tensor of size [batch_size, feature_dimension, len_all_featues]

Output:
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

class PredictionLayer(nn.Module):
    def __init__(self, args):
    
    def forward(self, args):