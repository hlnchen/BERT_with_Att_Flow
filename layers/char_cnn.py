"""
This file contains the definition of the character-level CNN layer.

The CNN should maps each word to a vector space.
The implementation can be based on:
1. https://arxiv.org/abs/1408.5882 (what BiDAF used)
2. https://arxiv.org/abs/1509.01626 (another high citation paper)
3. other reasonable choice of CNN

Input: 
the batch of [input_text], 
tensor of size [batch_size, token_dimension, length_text]

Output: the batch of [input_CNN_features]
tensor of size [batch_size, feature_dimension, length_text]

NOTE: 
1. CNN seems to require fixed-length inputs. Maybe need to set a max length.
2. This could be an optional scheme of our model: test if this helps and report
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CharCNN(nn.Module):
    def __init__(self, args):
        # Under construction
    
    def forward(self, args):