"""
Tweet success prediction
Tweet example
{
    "date": "2011-08-02 18:07:48",
    "device": "TweetDeck",
    "favorites": 49,
    "id": 98454970654916600,
    "isDeleted": "f",
    "isRetweet": "f",
    "retweets": 255,
    "text": "Republicans and Democrats have both created our economic problems."
}
"""
import os
import sys
import numpy as np
import json
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

from tweet_dataset import TweetDataset
from tweet_model import TweetPredictor

EPOCHS = 100
MAX_LEN = 10000
TRAIN_TEST_SPLIT = .2
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
OUTPUT_SIZE = 1
HIDDEN_SIZE = 256
EMBEDDING_LENGTH = 50
GLOVE_PATH = f'./glove/glove.6B.{EMBEDDING_LENGTH}d.txt'
TWEET_PATH = './trump_tweets.json'
# TEXT_MAX_LEN = 200

parser = argparse.ArgumentParser(description='Model trainer')
args = parser.parse_args()
args.file_path = TWEET_PATH

DEVICE = 'cpu'
if torch.cuda.is_available() and args.is_cuda:
    DEVICE = 'cuda'
    MAX_LEN = 0


def json_pretty_print(s):
    return json.dumps(s, indent=4, sort_keys=True)


def shuffle_and_split(dataset_size, split):
    indices = list(range(dataset_size))
    split = int(np.floor(split * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    return indices[split:], indices[:split]


# def normalize(t):
#     t: Tensor
#     tmax = t.max()
#     tmin = t.min()
#     if tmax == 0.0 and tmin == 0.0:
#         return t
#     return (t - tmin) / (tmax - tmin)


def main():
    # Since our dataset is from one file and will be used both for train and test loader,
    # create sampler which manually shuffle and split indices
    # dataset = TweetDataset(file_path=args.file_path, glove_path=GLOVE_PATH, max_len=MAX_LEN, text_len=TEXT_MAX_LEN)
    dataset = TweetDataset(file_path=args.file_path, glove_path=GLOVE_PATH, max_len=MAX_LEN)
    train_indices, test_indices = shuffle_and_split(
        dataset_size=len(dataset),
        split=TRAIN_TEST_SPLIT)

    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=SubsetRandomSampler(train_indices),
        batch_size=BATCH_SIZE,
    )

    # to not shuffle on test, we dont  instantiate SubsetRandomSampler
    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=SubsetRandomSampler(train_indices),
        # sampler=Sampler(test_indices),
        batch_size=BATCH_SIZE
    )

    model = TweetPredictor(
        batch_size=BATCH_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(dataset.vocabulary),
        embedding_length=EMBEDDING_LENGTH,
        weights=torch.FloatTensor(dataset.weights))
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    metrics = {}
    for stage in ['train', 'test']:
        for metric in ['loss']:
            # for metric in ['loss', 'acc']:
            metrics[f'{stage}_{metric}'] = []

    for epoch in range(EPOCHS):
        metrics_epoch = {key: [] for key in metrics.keys()}
        for data_loader in [data_loader_train, data_loader_test]:
            stage = 'train'
            torch.set_grad_enabled(True)
            if data_loader == data_loader_test:
                stage = 'test'
                torch.set_grad_enabled(False)

            # inference
            for x, y in data_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y_prim = model.forward(x)
                # TODO: bad loss
                # loss = torch.mean((normalize(y.float()) - normalize(y_prim)) ** 2)
                loss = torch.mean((y.float() - y_prim) ** 2)
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())  # Tensor(0.1) => 0.1f
                print(f'batch: epoch-{epoch} {loss.cpu().item()}  y max {y.max()} y_prim max {y_prim.max()}')

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')


if __name__ == '__main__':
    main()

# TODO: tokenize with Moses Perl https://github.com/moses-smt/mosesdecoder

# TODO: divide into words or smth?
# TODO: Tokenizer get cleaned texts
# TODO: change Tokenizer (from nltk.tokenize import sent_tokenize, word_tokenize)

# TODO: add save of model
# TODO: add accuaracy
# TODO: add metadata later (inputs: text, isretweet, deleted, day_of_week, retweet)
# TODO: add Tensorboard
# TODO: torch summary

# TODO: uninstall tensorflow, keras
# TODO: uninstall torchnlp
