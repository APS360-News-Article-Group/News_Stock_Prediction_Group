# all imports
import numpy as np
import time
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import json
import torchtext
import torchtext.data as data
import torchtext.vocab as vocab
from nltk.tokenize import RegexpTokenizer
import string

import random

# for padding
from torch.nn.utils.rnn import pad_sequence

import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

from spacy.lang.en import English


def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data


class TweetBatcher:
    def __init__(self, tweets, batch_size, drop_last=False):
        # store tweets by length
        self.tweets_by_length = {}
        for words, label in tweets:
            # compute the length of the tweet
            wlen = words.shape[0]
            # put the tweet in the correct key inside self.tweet_by_length
            if wlen not in self.tweets_by_length:
                self.tweets_by_length[wlen] = []
            self.tweets_by_length[wlen].append((words, label),)

        #  create a DataLoader for each set of tweets of the same length
        self.loaders = {wlen: torch.utils.data.DataLoader(
            tweets,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last)  # omit last batch if smaller than batch_size
            for wlen, tweets in self.tweets_by_length.items()}

    def __iter__(self):  # called by Python to create an iterator
        # make an iterator for every tweet length
        iters = [iter(loader) for loader in self.loaders.values()]
        while iters:
            # pick an iterator (a length)
            im = random.choice(iters)
            try:
                yield next(im)
            except StopIteration:
                # no more elements in the iterator, remove it
                iters.remove(im)


def dataSplit(wordLimit):
    print("Datasplit intialized.")
    # first, load the stock json obtained from stock_data_crawler
    train, valid, test = [], [], []
    fileLoc = "C:\\Temp\\finalData_big_balanced.json"
    modelLoc = "C:\\Temp\\GoogleNews-vectors-negative300.bin.gz"
    # wordLimit = 1000000
    # wordLimit = 5000
    stockJson = loadJson(fileLoc)
    model = gensim.models.KeyedVectors.load_word2vec_format(
        modelLoc, binary=True, limit=wordLimit)

    # tokenizer = English()
    tokenizer = RegexpTokenizer(r'\w+')

    model_itos = model.index2word
    model_stoi = model.vocab
    errorList = []

    for i, item in enumerate(stockJson):
        # we need to tokenize this sentence
        # item['title']
        try:
            # token_list = [word.text for word in tokenizer(item['title'] + item['description'])]
            # token_list = [word for word in tokenizer.tokenize(
            #     item['headline'])]

            token_list = tokenizer.tokenize(item['headline'])

            # token_list = [word.lower() for word in token_list]
            # model_stoi = {"'m":1, "to":2, "our":8}

            idxs = [
                model_stoi[word].index for word in token_list if word in model_stoi]
            idxs = torch.tensor(idxs)
            label = torch.tensor(int(item['label'])).long()
            if i % 10 < 9:
                train.append((idxs, label))
            else:
                valid.append((idxs, label))
        except:
            errorList.append(item)

    i2, j2, k2 = len(train), len(valid), len(test)
    # model_emb = nn.Embedding.from_pretrained(model.vectors)
    print("Datasplit complete.")
    return train, valid, test, model


def get_accuracy(model, data_loader):
    correct, total = 0, 0
    errLog = []
    for tweets, labels in data_loader:
        try:
            output = model(tweets)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.shape[0]
        except:
            errLog.append((tweets, labels))

    return correct / total


class NewsRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embedding):
        super(NewsRNN, self).__init__()
        self.emb = embedding

        # 300 since vector is of size 300
        # self.emb = torch.eye(300)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc1 = nn.Linear(hidden_size, 80)
        self.fc2 = nn.Linear(80, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])

        return out


class NewsGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embedding):
        super(NewsGRU, self).__init__()
        self.emb = embedding

        # 300 since vector is of size 300
        # self.emb = torch.eye(300)
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.rnn = nn.GRU(input_size, hidden_size, num_layers = 2, dropout = 0.1, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc1 = nn.Linear(hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x, (h0, c0))
        # Pass the output of the last time step to the classifier
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)

        return out


class NewsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embedding):
        super(NewsLSTM, self).__init__()
        self.emb = embedding

        # 300 since vector is of size 300
        # self.emb = torch.eye(300)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers, dropout = 0.07, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x, (h0, c0))
        # Pass the output of the last time step to the classifier
        out = self.relu(self.fc_1(out[:, -1, :]))
        out = self.fc_2(out)
        # out = self.relu(self.fc_3(out))
        # out = self.fc_4(out)

        # out = self.fc(out[:, -1, :])

        return out


def train_rnn_network(model, train_loader, valid_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    errLog = []
    for epoch in range(num_epochs):
        for tweets, labels in train_loader:
            try:
                optimizer.zero_grad()
                pred = model(tweets)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
            except:
                errLog.append((tweets, labels))
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train_loader))
        valid_acc.append(get_accuracy(model, valid_loader))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1]))
    # plotting
    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, valid_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()


def mainLoop():
    # split data, wordLimit = take word vectors of the 'n' most commonly used words in GoogleNews
    train, valid, test, model = dataSplit(wordLimit=50000)

    # define weights and embedding of pretrained word2vec model
    weights = torch.FloatTensor(model.vectors)
    embedding = nn.Embedding.from_pretrained(weights)

    # load data, need to work on test loader
    train_loader = TweetBatcher(train, batch_size=248, drop_last=False)
    valid_loader = TweetBatcher(valid, batch_size=248, drop_last=False)

    # valina RNN and LSTM
    model_simple = NewsRNN(300, 315, 2, embedding)
    # model_medium = NewsGRU(300, 2048, 2, embedding)
    model_complex = NewsLSTM(300, 555, 2, 2, embedding)

    # get initial accuracy
    # print("Get accuracy initialized.")
    # print(get_accuracy(model_complex, train_loader))
    # print("Get accuracy complete.")

    # train network
    train_rnn_network(model_complex, train_loader, valid_loader,
                      num_epochs=50, learning_rate=1e-4)


mainLoop()

print("done")
