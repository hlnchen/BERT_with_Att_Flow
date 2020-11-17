""" 
This file contains the definition of the attention-flow layer based on BiDAF.

Input: 
the batch of [question_features, context_features];
the input_features is obtained by concatenating BERT and CNN(optional) features;
tensor of size: [batch_size, feature_dimension, len_q_feqtures + len_c_features]

Output: the batch of [attention_features];
tensor of size: [batch_size, feature_dimension, len_q_feqtures + len_c_features]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttFlow(nn.Module):
    def __init__(self, args):
        # Under construction
    
    def forward(self, args):