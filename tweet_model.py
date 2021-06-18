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


def sort_sequences(inputs, lengths):
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()
    return inputs[sorted_idx], lengths_sorted, unsorted_idx


class TweetPredictor(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        """
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
        self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # input mean, max, first, last output for each sequence of batch.
        # Since LSTM is bidirectional, output dimensions are 2*hidden_size
        self.fc = nn.Linear(hidden_size * 4 * 2, output_size)

    def forward(self, x, real_len):
        """
        x shape (batch_size, num_sequences)
        real_len shape (batch_size, 1)
        """
        x = self.word_embeddings(x)

        # packing
        # x, real_len, _ = sort_sequences(x, real_len)
        # x = torch.nn.utils.rnn.pack_padded_sequence(input=x, lengths=real_len, batch_first=True)
        # x shape (batch_size, num_sequences, embedding_length)

        # h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        # c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        # output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))

        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        # output shape (batch_size, embedding_length, 2 * hidden_size)
        # final_hidden_state and final_cell_state shape (? 2 (bi-directional), batch_size, hidden_size)

        # TODO: Huber loss
        # TODO: concat multimodal (text props: isretweet, deleted, day_of_week, retweet)
        out = None
        batch_size = x.shape[0]
        for b in range(batch_size):
            seq_out = torch.cat((
                output[b, 0],
                output[b, -1],
                torch.mean(output[b], dim=0),
                torch.max(output[b], dim=0)[0]
            ), dim=0)
            seq_out = seq_out.clone().detach().reshape((1, -1))
            if out is None:
                out = seq_out
            else:
                out = torch.cat((out, seq_out))
        out = self.fc(out)
        # out = self.fc(final_hidden_state[-1])
        out = nn.ReLU()(out)
        # out shape (batch_size, output_size)
        return out

# LSTM(128, return_seq=True) -> Dropout(0.6) -> LSTM(128, return_seq=True)
# -> Dropout(0.6) -> LSTM(128) -> Dense(1, sigmoid)
