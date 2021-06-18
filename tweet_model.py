import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import os
import sys
import numpy as np

import string
import keras
from sklearn.model_selection import train_test_split
import json


class TweetPredictor(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        """
        Arguments
        ---------
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        """
        super(TweetPredictor, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x shape (batch_size, num_sequences)
        """
        x = self.word_embeddings(x)
        x = x.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)

        # batch_size = x.shape[0]
        # h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        # c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        # output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        output, (final_hidden_state, final_cell_state) = self.lstm(x)

        # final_hidden_state.size() = (batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        out = self.fc(final_hidden_state[-1])
        out = self.relu(out)
        return out

# LSTM(128, return_seq=True) -> Dropout(0.6) -> LSTM(128, return_seq=True)
# -> Dropout(0.6) -> LSTM(128) -> Dense(1, sigmoid)
