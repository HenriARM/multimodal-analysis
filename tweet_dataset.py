from torch.utils.data import Dataset
import keras
from keras.preprocessing.text import Tokenizer
import json
import numpy as np
import torch
import string
import emoji

import utils


def load_glove(glove_path):
    with open(glove_path, 'r', encoding='UTF-8') as f:
        vectors = {}
        for s in f:
            line = s.split()
            vectors[line[0]] = np.array(line[1:], dtype=np.float64)
    return vectors


def clean_text(s):
    # tweets don't consist of html tags so no need to remove them
    # remove punctuation
    for punc in string.punctuation:
        s = s.replace(punc, '')

    # remove stop words
    l = []
    for word in s.split():
        if word not in utils.stopwords:
            l.append(word)
    s = ' '.join(l)

    # remove emoji
    l = []
    for c in s:
        if c not in emoji.UNICODE_EMOJI:
            l.append(c)
        else:
            print('True')
            exit(-3)
    s = ' '.join(l)

    return s


class TweetDataset(Dataset):
    # def __init__(self, file_path, glove_path, max_len, text_len):
    def __init__(self, file_path, glove_path, max_len):
        super().__init__()

        # Load tweets
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.max_len = max_len

        # create vocabulary
        texts = []
        for tweet in self.data:
            texts.append(clean_text(tweet['text']))
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(texts)
        self.vocabulary = self.tokenizer.word_index

        # load GloVe word2vec (dict: word -> vector)
        self.word2vec = load_glove(glove_path=glove_path)

        # create weights matrix
        self.weights = np.zeros((len(self.vocabulary), self.word2vec['the'].shape[0]))
        for word, index in self.vocabulary.items():
            embedding = self.word2vec.get(word)
            if embedding is not None:
                self.weights[index, :] = embedding

        # self.text_len = text_len

    def __len__(self):
        if self.max_len:
            return self.max_len
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        text = clean_text(text)
        text = self.tokenizer.texts_to_sequences(text)
        # flatten
        text = [i for s in text for i in s]
        # # pad with zeroes
        # text = np.asarray(text)
        # text_cliped = text[:self.text_len].copy()
        # text_pad = np.zeros(self.text_len)
        # text_pad[:text_cliped.shape[0]] = text_cliped
        # return torch.LongTensor(text_pad), torch.LongTensor([self.data[idx]['favorites']])
        return torch.LongTensor(text), torch.LongTensor([self.data[idx]['favorites']])
