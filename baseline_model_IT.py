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
import tensorflow_hub as hub
import tensorflow as tf
import dataSplitPkg

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


# for this datasplit, words are not tokenized nor converted to indices
def sent2vec_dataSplit():
    targetComp = "ACN ATVI ADBE AMD AKAM ADS GOOGL GOOG APH ADI ANSS AAPL AMAT ADSK ADP AVGO CA CDNS CSCO CTXS CTSH GLW CSRA DXC EBAY EA FFIV FB FIS FISV FLIR IT GPN HRS HPE HPQ INTC IBM INTU IPGP JNPR KLAC LRCX MA MCHP MU MSFT MSI NTAP NFLX NVDA ORCL PAYX PYPL QRVO QCOM RHT CRM STX SWKS SYMC SNPS TTWO TEL TXN TSS VRSN V WDC WU XRX XLNX"
    targetCompList = targetComp.split(" ")
    print("Datasplit intialized.")
    # first, load the stock json obtained from stock_data_crawler
    train, valid, test, errList = [], [], [], []
    fileLoc = "C:\\Temp\\finalData_big_balanced.json"
    modelLoc = "C:\\Temp\\GoogleNews-vectors-negative300.bin.gz"
    stockJson = loadJson(fileLoc)

    for i, item in enumerate(stockJson):
        # we need to tokenize this sentence
        # item['title']
        if item['companySymbol'] in targetCompList:
            try:
                if i % 10 < 8:
                    train.append((item['headline'], item['label']))
                else:
                    valid.append((item['headline'], item['label']))
            except:
                errList.append(item)

    print("Datasplit complete.")
    return train, valid, test

def word2vec_dataSplit(wordLimit):
    print("Datasplit intialized.")

    # first, load the stock json obtained from stock_data_crawler
    train, valid, test = [], [], []
    fileLoc = "finaldata_half.json"
    modelLoc = "C:\\Temp\\GoogleNews-vectors-negative300.bin.gz"

    trainJson, validJson, testJson = dataSplitPkg.mainLoop(fileLoc=fileLoc)

    model = gensim.models.KeyedVectors.load_word2vec_format(
        modelLoc, binary=True, limit=wordLimit)

    # tokenizer = English()
    tokenizer = RegexpTokenizer(r'\w+')

    model_itos = model.index2word
    model_stoi = model.vocab
    errorList = []
    
    switchData = True

    for i, item in enumerate(trainJson):
        try:
            token_list = tokenizer.tokenize(item['headline'])
            token_list_1 = [word for word in token_list]

            idxs = [
                model_stoi[word].index for word in token_list_1 if word in model_stoi]
            idxs = torch.tensor(idxs)
            label = torch.tensor(int(item['label'])).long()

            train.append((idxs, label))  

        except:
            errorList.append(item)
    
    for i, item in enumerate(validJson):
        try:
            token_list = tokenizer.tokenize(item['headline'])
            token_list_1 = [word for word in token_list]

            idxs = [
                model_stoi[word].index for word in token_list_1 if word in model_stoi]
            idxs = torch.tensor(idxs)
            label = torch.tensor(int(item['label'])).long()

            valid.append((idxs, label))  

        except:
            errorList.append(item)

    for i, item in enumerate(testJson):
        try:
            token_list = tokenizer.tokenize(item['headline'])
            token_list_1 = [word for word in token_list]

            idxs = [
                model_stoi[word].index for word in token_list_1 if word in model_stoi]
            idxs = torch.tensor(idxs)
            label = torch.tensor(int(item['label'])).long()

            test.append((idxs, label))  

        except:
            errorList.append(item)

    i2, j2, k2 = len(train), len(valid), len(test)
    # model_emb = nn.Embedding.from_pretrained(model.vectors)

    test_allPositive = []
    test_allNegative = []

    for item in test:
        if item[1] == 1:
            test_allPositive.append(item)
        else:
            test_allNegative.append(item)

    print("Datasplit complete.")
    return train, valid, test, model, test_allPositive, test_allNegative


# for this datasplit, words are not tokenized nor converted to indices
def sent2vec_dataSplit():

    targetComp = "ACN ATVI ADBE AMD AKAM ADS GOOGL GOOG APH ADI ANSS AAPL AMAT ADSK ADP AVGO CA CDNS CSCO CTXS CTSH GLW CSRA DXC EBAY EA FFIV FB FIS FISV FLIR IT GPN HRS HPE HPQ INTC IBM INTU IPGP JNPR KLAC LRCX MA MCHP MU MSFT MSI NTAP NFLX NVDA ORCL PAYX PYPL QRVO QCOM RHT CRM STX SWKS SYMC SNPS TTWO TEL TXN TSS VRSN V WDC WU XRX XLNX"
    targetCompList = targetComp.split(" ")
    print("Datasplit intialized.")
    # first, load the stock json obtained from stock_data_crawler
    train, valid, test, errList = [], [], [], []
    fileLoc = "C:\\Temp\\finalData_big_balanced.json"
    modelLoc = "C:\\Temp\\GoogleNews-vectors-negative300.bin.gz"
    # wordLimit = 1000000
    # wordLimit = 5000
    stockJson = loadJson(fileLoc)

    for i, item in enumerate(stockJson):
        # we need to tokenize this sentence
        # item['title']
        if item['companySymbol'] in targetCompList:
            try:
                if i % 10 < 8:
                    train.append((item['headline'], item['label']))
                else:
                    valid.append((item['headline'], item['label']))
            except:
                errList.append(item)

    print("Datasplit complete.")
    return train, valid, test


def get_accuracy(model, data_loader):
    correct, total = 0, 0
    precision_correct, precision_total = 0, 0
    errLog = []

    for batch in data_loader:
        try:
            output = model(batch[0])
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(batch[1].view_as(pred)).sum().item()
            total += batch[1].shape[0]

            # precision calculation
            indices = [i for i in range(pred.shape[0]) if pred[i] == 1]
            for j in indices:
                if batch[1][j] == 1:
                    precision_correct += 1
            precision_total += len(indices)

        except:
            errLog.append((batch[0], batch[1]))
    
    if precision_total == 0:
        ha = 0
    else:
        ha = precision_correct / precision_total

    return correct / total, ha


class doc2vec_baselineModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(doc2vec_baselineModel, self).__init__()
        self.emb = embed_useT("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.relu = nn.ReLU()

        # 150 --> 50 --> 2
        self.fc_1 = nn.Linear(input_size, 1024)
        self.fc_2 = nn.Linear(1024, 300)
        self.fc_3 = nn.Linear(300, 150)
        self.fc_4 = nn.Linear(150, num_classes)
        
    def forward(self, x):
        # Look up the embedding
        x_emb = self.emb(x)

        # 2 simple fc layers
        out = self.relu(self.fc_1(x_emb))
        out = self.relu(self.fc_2(out))
        out = self.relu(self.fc_3(out))
        out = self.fc_4(out)

        return out

class baselineModel(nn.Module):
    def __init__(self, input_size, num_classes, embedding):
        super(baselineModel, self).__init__()
        self.emb = embedding
        self.name = "baseline"
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # 150 --> 50 --> 2
        self.fc_1 = nn.Linear(input_size, 190)
        self.fc_2 = nn.Linear(190, 120)
        self.fc_3 = nn.Linear(120, 60)
        self.fc_4 = nn.Linear(60, num_classes)

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)

        # sum the BOW tensors into one SENT tensor
        x_1 = torch.sum(x, dim=1)

        # 2 simple fc layers
        out = self.dropout(self.relu(self.fc_1(x_1)))
        out = self.dropout(self.relu(self.fc_2(out)))
        out = self.dropout(self.relu(self.fc_3(out)))
        out = self.fc_4(out)

        return out


def train_rnn_network(model, train_iter, valid_iter, num_epochs, learning_rate, batch_size, hidden_size):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc, train_prec, valid_prec = [], [], [], [], []
    best_model_acc = 0
    epochs = []
    errLog = []

    for epoch in range(num_epochs):
        model_path = "C:\\Temp\\best_models\\baseline_models\\" + F'name{model.name}_bs{batch_size}_hs{hidden_size}_lr{learning_rate}_epoch{epoch+1}.pth' 
            
        for tweets, labels in train_iter:
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
        train_acc_value, train_prec_value = get_accuracy(model, train_iter)
        valid_acc_value, valid_prec_value = get_accuracy(model, valid_iter)
        train_acc.append(train_acc_value)
        valid_acc.append(valid_acc_value)
        train_prec.append(train_prec_value)
        valid_prec.append(valid_prec_value)

        if valid_acc[-1] > best_model_acc:
            best_model_acc = valid_acc[-1]
            best_epoch = epoch + 1

            # save your state_dict model
            torch.save(model.state_dict(), model_path)

        print("Epoch %d; Loss %f; Best Epoch %d; Train Acc %f; Train Precision %f; Val Acc %f; Val Precision %f" % (
              epoch+1, loss, best_epoch, train_acc[-1], train_prec[-1], valid_acc[-1], valid_prec[-1]))

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

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: torch.from_numpy(session.run(embeddings, {sentences: x}))

def mainLoop_sent2vec():
    # split data, wordLimit = take word vectors of the 'n' most commonly used words in GoogleNews
    train, valid, test = sent2vec_dataSplit()

    # load data, need to work on test loader 248 before
    train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
    valid_loader  = torch.utils.data.DataLoader(valid, batch_size = 64, shuffle = True)
    
    # baseline model
    model_baseline_doc2vec = doc2vec_baselineModel(512, 2)

    # get initial accuracy
    print("Get accuracy initialized.")
    print(get_accuracy(model_baseline_doc2vec, train_loader))
    print("Get accuracy complete.")

    # train network
    train_rnn_network(model_baseline_doc2vec, train_loader, valid_loader,
                      num_epochs=90, learning_rate=1.2e-4)

def mainLoop_word2vec():
    # split data, wordLimit = take word vectors of the 'n' most commonly used words in GoogleNews
    train, valid, test, model, test_allPositive, test_allNegative = word2vec_dataSplit(wordLimit=100000)

    # define weights and embedding of pretrained word2vec model
    weights = torch.FloatTensor(model.vectors)
    embedding = nn.Embedding.from_pretrained(weights)

    # initial hyper parameter definition
    batch_size = 32
    hidden_size = "N/A"

    # load data, need to work on test loader 248 before
    train_loader = TweetBatcher(train, batch_size = batch_size, drop_last = False)
    valid_loader = TweetBatcher(valid, batch_size = batch_size, drop_last = False)
    
    # baseline model
    model_baseline_fc = baselineModel(300, 2, embedding)

    # get initial accuracy
    print("Get accuracy initialized.")
    print(get_accuracy(model_baseline_fc, train_loader))
    print("Get accuracy complete.")

    # train network
    train_rnn_network(model_baseline_fc, train_loader, valid_loader,
                      num_epochs=150, learning_rate=1.2e-4, batch_size=batch_size, hidden_size="na")

# mainLoop_sent2vec()

def test_word2vec(model_path):
    # split data, wordLimit = take word vectors of the 'n' most commonly used words in GoogleNews
    train, valid, test, model, test_allPositive, test_allNegative = word2vec_dataSplit(wordLimit=100000)

    # initial hyper parameter definition
    batch_size = 32
    hidden_size = "N/A"

    # define weights and embedding of pretrained word2vec model
    weights = torch.FloatTensor(model.vectors)
    embedding = nn.Embedding.from_pretrained(weights)

    model_test = baselineModel(300, 2, embedding)
    model_test.load_state_dict(torch.load(model_path))

    valid_loader = TweetBatcher(valid, batch_size = batch_size, drop_last = False)
    test_loader = TweetBatcher(test, batch_size = batch_size, drop_last = False)

    test_loader_pos = TweetBatcher(test_allPositive, batch_size = batch_size, drop_last = False)
    test_loader_neg = TweetBatcher(test_allNegative, batch_size = batch_size, drop_last = False)

    print("All pos.")
    print(get_accuracy(model_test, test_loader_pos))
    print("All neg.")
    print(get_accuracy(model_test, test_loader_neg))

mainLoop_word2vec()

# test_word2vec(model_path = "C:\\Temp\\best_models\\baseline_models\\namebaseline_bs64_hsna_lr0.00012_epoch98.pth")

print("done")
