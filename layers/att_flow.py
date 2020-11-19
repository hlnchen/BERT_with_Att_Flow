""" 
This file contains the definition of the attention-flow layer based on BiDAF.

Input: 
The batch of [context_features] and [question_features];
the features are obtained by concatenating BERT and CNN(optional) features;
tensor of size: [batch_size, length_context, feature_dimension], [batch_size, length_quesiton feature_dimension]

Output: the batch of [attention_features];
tensor of size: [batch_size, len_q_feqtures + len_c_features, feature_dimension]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttFlow(nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        self.weight = nn.Linear(3*feature_dimension, 1, bias=False)

    def forward(self, context_features: torch.Tensor, question_features: torch.Tensor):
        # Construct a similarity matrix
        batch_size = context_features.shape[0]          # N
        feature_dimension = context_features.shape[2]   # d
        length_context = context_features.shape[1]      # T
        length_quesiton = question_features.shape[1]    # J
        shape = (batch_size, length_context, length_quesiton, feature_dimension)    # (N,T,J,d)
        context_features_expanded = context_features.unsqueeze(2)                   # (N,T,1,d)
        context_features_expanded = context_features_expanded.expand(shape)         # (N,d,T,J)
        question_features_expanded = question_features.unsqueeze(1)                 # (N,1,J,d)
        question_features_expanded = question_features_expanded.expand(shape)       # (N,T,J,d)
        entrywise_prod = torch.mul(context_features_expanded, question_features_expanded)   # (N,T,J,d)
        concat_feature = torch.cat((context_features_expanded, question_features_expanded, entrywise_prod), 2) # (N,T,J,3d)
        similarity = self.weight(concat_feature).view(batch_size, length_context, length_quesiton) # (N,T,J)

        # Context2Question attention
        c2q = torch.bmm(F.softmax(similarity, dim=2), question_features) # (N,T,J) * (N,J,d)

        # Question2Context attetion
        b = F.softmax(torch.max(similarity, dim=2)[0], dim=-1) # (N,T)
        q2c = torch.bmm(b.unsqueeze(1), context_features) # (N,1,T) * (N,T,d)
        q2c = q2c.repeat(1,length_context,1)

        return c2q, q2c