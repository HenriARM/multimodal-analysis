from torch.utils.data import Dataset
import keras
from keras.preprocessing.text import Tokenizer
import json
import numpy as np
import torch
import string
import emoji
import re

import utils


def load_glove(glove_path):
    with open(glove_path, 'r', encoding='UTF-8') as f:
        vectors = {}
        for s in f:
            line = s.split()
            vectors[line[0]] = np.array(line[1:], dtype=np.float64)
    return vectors


def clean_text(s):
    # ############# Twitter specific cleaning ############
    # this tweets don't consist of html tags so no need to remove them

    # remove non-ascii chars
    s = ''.join(filter(lambda x: x in string.printable, s))

    # TODO: make so tokenizer don't tokenize '@user' in smaller parts
    # remove words with mentions
    s = re.sub('(@)[A-Za-z0-9]+', '@user ', s)

    # TODO: remove mentions which are at the start and end of sentence

    # remove hashtag sign but keep the text
    s.replace('#', ' ').replace('_', ' ')

    # # remove RT (retweet)
    # s = s.replace('RT', ' ')

    # # remove links
    s = re.sub(r'(http|www)\S+', '@url', s)

    # change all words to lower case (too many uppercase or alpha words)
    s = s.lower()

    # ############# Twitter specific cleaning ############

    # # remove punctuation
    # for punc in string.punctuation:
    #     s = s.replace(punc, ' ')

    # # remove stop words
    # l = []
    # for word in s.split():
    #     if word not in utils.stopwords:
    #         l.append(word)
    # s = ' '.join(l)

    # # remove emoji
    # l = []
    # for c in s:
    #     if c not in emoji.UNICODE_EMOJI['en']:
    #         l.append(c)
    # s = ''.join(l)

    # remove excess spaces in strings
    s = re.sub(' +', ' ', s)
    return s


class TweetDataset(Dataset):
    def __init__(self, file_path, glove_path, max_len, text_len):
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
        self.weights = np.zeros((len(self.vocabulary) + 1, self.word2vec['the'].shape[0]))
        for word, index in self.vocabulary.items():
            embedding = self.word2vec.get(word)
            if embedding is not None:
                self.weights[index, :] = embedding

        self.text_len = text_len

    def __len__(self):
        if self.max_len:
            return self.max_len
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        text = clean_text(text)
        clean_text_pad = text
        text = self.tokenizer.texts_to_sequences(text)
        # flatten
        text = [i for s in text for i in s]
        # pad with zeroes
        text = np.asarray(text)
        text_cliped = text[:self.text_len].copy()
        text_pad = np.zeros(self.text_len)
        text_pad[:text_cliped.shape[0]] = text_cliped
        return torch.LongTensor(text_pad), torch.LongTensor([self.data[idx]['favorites']]), clean_text_pad
