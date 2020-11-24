"""
This file contains the definition of the character-level CNN layer.

The CNN should maps each word to a vector space.
The implementation can be based on:
https://arxiv.org/abs/1509.01626

Input: 
the batch of [input_ids], 
tensor of size [batch_size, sequence_length]
NOTE: the input_ids is the same as what used for BERT.

Output: the batch of [input_CNN_features]
tensor of size [batch_size, feature_dimension, sequence_length]

NOTE: This could be an optional scheme of our model: test if this helps and report
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CharCNN(nn.Module):
    def __init__(self, input_length, vocab_size):
        # Under construction
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim = vocab_size)
        self.conv1 = nn.Sequential(nn.Conv2d(1,1,7), nn.MaxPool2d(3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(1,1,3), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(1,1,3), nn.MaxPool2d(3), nn.ReLU())
        self.cnn = nn.Sequential(
            conv1,conv1,conv2,conv2,conv2,conv3, 
            nn.Linear(in_features= None,out_features=None), 
            nn.Softmax())
    def forward(self, cnn_input_ids):
        embed = self.embedding_layer(cnn_input_ids)
        cnn_feature = self.cnn(embed)
        return cnn_feature


# # ===========================================================================

# '''
# Character level CNN implementation(Sample)
# '''


# import numpy as np
# import pandas as pd
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# from keras.layers import Input, Embedding, Activation, Flatten, Dense
# from keras.layers import Conv1D, MaxPooling1D, Dropout
# from keras.models import Model

# '''
# Load data
# '''
# train_data_source = './train.csv'
# test_data_source = './test.csv'

# train_df = pd.read_csv(train_data_source, header=None)
# test_df = pd.read_csv(test_data_source, header=None)

# # concatenate column 1 and column 2 as one text
# for df in [train_df, test_df]:
#     df[1] = df[1] + df[2]
#     df = df.drop([2], axis=1)

# # convert string to lower case
# train_texts = train_df[1].values
# train_texts = [s.lower() for s in train_texts]

# test_texts = test_df[1].values
# test_texts = [s.lower() for s in test_texts]

# '''Convert string to index'''

# # Tokenizer
# tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
# tk.fit_on_texts(train_texts)
# # If we already have a character list, then replace the tk.word_index
# # If not, just skip below part

# # -----------------------Skip part start--------------------------
# # construct a new vocabulary
# alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
# char_dict = {}
# for i, char in enumerate(alphabet):
#     char_dict[char] = i + 1

# # Use char_dict to replace the tk.word_index
# tk.word_index = char_dict.copy()
# # Add 'UNK' to the vocabulary
# tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
# # -----------------------Skip part end----------------------------

# # Convert string to index
# train_sequences = tk.texts_to_sequences(train_texts)
# test_texts = tk.texts_to_sequences(test_texts)

# # Padding
# train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
# test_data = pad_sequences(test_texts, maxlen=1014, padding='post')

# # Convert to numpy array
# train_data = np.array(train_data, dtype='float32')
# test_data = np.array(test_data, dtype='float32')


# '''Get classes'''
# train_classes = train_df[0].values
# train_class_list = [x - 1 for x in train_classes]

# test_classes = test_df[0].values
# test_class_list = [x - 1 for x in test_classes]

# from keras.utils import to_categorical

# train_classes = to_categorical(train_class_list)
# test_classes = to_categorical(test_class_list)


# # =====================Char CNN=======================
# # parameter
# input_size = 1014
# vocab_size = len(tk.word_index)
# embedding_size = 69
# conv_layers = [[256, 7, 3],
#                [256, 7, 3],
#                [256, 3, -1],
#                [256, 3, -1],
#                [256, 3, -1],
#                [256, 3, 3]]

# fully_connected_layers = [1024, 1024]
# num_of_classes = 4
# dropout_p = 0.5
# optimizer = 'adam'
# loss = 'categorical_crossentropy'

# # Embedding weights
# embedding_weights = []  # (70, 69)
# embedding_weights.append(np.zeros(vocab_size))  # (0, 69)

# for char, i in tk.word_index.items():  # from index 1 to 69
#     onehot = np.zeros(vocab_size)
#     onehot[i - 1] = 1
#     embedding_weights.append(onehot)

# embedding_weights = np.array(embedding_weights)
# print('Load')

# # Embedding layer Initialization
# embedding_layer = Embedding(vocab_size + 1,
#                             embedding_size,
#                             input_length=input_size,
#                             weights=[embedding_weights])

# # Model Construction
# # Input
# inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
# # Embedding
# x = embedding_layer(inputs)
# # Conv
# for filter_num, filter_size, pooling_size in conv_layers:
#     x = Conv1D(filter_num, filter_size)(x)
#     x = Activation('relu')(x)
#     if pooling_size != -1:
#         x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
# x = Flatten()(x)  # (None, 8704)
# # Fully connected layers
# for dense_size in fully_connected_layers:
#     x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
#     x = Dropout(dropout_p)(x)
# # Output Layer
# predictions = Dense(num_of_classes, activation='softmax')(x)
# # Build model
# model = Model(inputs=inputs, outputs=predictions)
# model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
# model.summary()

# # # 1000 training samples and 100 testing samples
# # indices = np.arange(train_data.shape[0])
# # np.random.shuffle(indices)
# #
# # x_train = train_data[indices][:1000]
# # y_train = train_classes[indices][:1000]
# #
# # x_test = test_data[:100]
# # y_test = test_classes[:100]

# indices = np.arange(train_data.shape[0])
# np.random.shuffle(indices)

# x_train = train_data[indices]
# y_train = train_classes[indices]

# x_test = test_data
# y_test = test_classes

# # Training
# model.fit(x_train, y_train,
#           validation_data=(x_test, y_test),
#           batch_size=128,
#           epochs=10,
#           verbose=2)
