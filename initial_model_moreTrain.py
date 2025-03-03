# all imports
import numpy as np
import time
import torch
import torch.utils.data
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
import tensorflow as tf
import nltk
from nltk.tokenize import RegexpTokenizer
import string
import torchtext
from torchtext.vocab import Vectors
from nltk.corpus import stopwords
from textblob import Word
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


def dataSplit(wordLimit):
    targetComp = "ACN ATVI ADBE AMD AKAM ADS GOOGL GOOG APH ADI ANSS AAPL AMAT ADSK ADP AVGO CA CDNS CSCO CTXS CTSH GLW CSRA DXC EBAY EA FFIV FB FIS FISV FLIR IT GPN HRS HPE HPQ INTC IBM INTU IPGP JNPR KLAC LRCX MA MCHP MU MSFT MSI NTAP NFLX NVDA ORCL PAYX PYPL QRVO QCOM RHT CRM STX SWKS SYMC SNPS TTWO TEL TXN TSS VRSN V WDC WU XRX XLNX"
    targetCompList = targetComp.split(" ")

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
        if item['companySymbol'] in targetCompList:
            try:
                # token_list = [word.text for word in tokenizer(item['title'] + item['description'])]
                # token_list = [word for word in tokenizer.tokenize(
                #     item['headline'])]

                token_list = tokenizer.tokenize(item['headline'])
                token_list_2 = [word.lower() for word in token_list]

                idxs = [model_stoi[word].index for word in token_list_2 if word in model_stoi]
                idxs = torch.tensor(idxs)
                label = torch.tensor(int(item['label'])).long()
                if i % 10 < 8:
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
    model.train = False
    correct, total = 0, 0
    precision_correct, precision_total = 0, 0
    errLog = []
    pred_total, label_total = [], []

    for batch in data_loader:
        output = model(batch.title)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(batch.changePercent.view_as(pred)).sum().item()
        total += batch.label.shape[0]
        pred_total += [int(i) for i in pred]
        label_total += [int(i) for i in batch.changePercent]

        # precision calculation
        indices = [i for i in range(pred.shape[0]) if pred[i] == 1]
        for j in indices:
            if batch.changePercent[j] == 1:
                precision_correct += 1
        precision_total += len(indices)

    confusion = tf.confusion_matrix(labels=label_total, predictions=pred_total)
    sess = tf.Session()
    with sess.as_default():
        print(sess.run(confusion))
    
    if precision_total == 0:
        ha = 0
    else:
        ha = precision_correct / precision_total

    return correct / total, ha


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
        x1 = self.emb(x[0])
        # Set an initial hidden state
        h0 = torch.zeros(1, x[0].size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x1, h0)
        # Pass the output of the last time step to the classifier
        # out = self.fc(out[:, torch.LongTensor(x[1]), :])
        out = self.fc(out[:, -1, :])

        return out


class NewsRNN_eye(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embedding):
        super(NewsRNN_eye, self).__init__()
        self.emb = torch.eye(input_size)
        # 300 since vector is of size 300
        # self.emb = torch.eye(300)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Look up the embedding
        x1 = self.emb[x[0]]
        # Set an initial hidden state
        h0 = torch.zeros(1, len(x1), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x1, h0)
        # Pass the output of the last time step to the classifier
        # out = self.fc(out[:, torch.LongTensor(x[1]), :])
        out = self.fc(out[:, -1, :])

        return out


class NewsGRU_eye(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NewsGRU_eye, self).__init__()
        self.emb = torch.eye(input_size)
        self.hidden_size = hidden_size
        self.name = "NewsGRU_eye"
        self.rnn = nn.GRU(input_size, hidden_size, dropout=0.2, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_1 = nn.Linear(hidden_size, 100)
        self.fc_2 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        a, b = x[0], x[1]
        # Look up the embedding
        x = self.emb[x[0]]
        # Set an initial hidden state
        h0 = torch.zeros(1, x.shape[0], self.hidden_size)
        # Forward propagate the GRU 
        out, _ = self.rnn(x, h0)
        # Pass the output of the last time step to the classifier
        one = out[:, b.long()-1, :]
        out = self.fc(one[:, -1, :])

        return out


class NewsGRU_word2vec(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embedding):
        super(NewsGRU_word2vec, self).__init__()
        self.emb = embedding
        self.name = "NewsGRU_word2vec"
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, dropout=0.2, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.fc_1 = nn.Linear(hidden_size, 100)
        self.fc_2 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        a, b = x[0], x[1]
        # Look up the embedding
        x = self.emb(x[0])
        # Set an initial hidden state
        h0 = torch.zeros(1, x.shape[0], self.hidden_size)
        # Forward propagate the GRU 
        out, _ = self.rnn(x, h0)
        # Pass the output of the last time step to the classifier
        one = out[:, b.long()-1, :]
        # out1 = self.fc(one[:, -1, :])

        # out2 = self.fc(torch.max(out, dim=1)[0])

        out2 = torch.cat([torch.max(out, dim=1)[0], 
                 torch.mean(out, dim=1)], dim=1)
        out2 = self.fc(out2)

        return out2

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
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Look up the embedding
        x1 = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x1, (h0, c0))
        # Pass the output of the last time step to the classifier
        out1 = self.fc_1(out[:, -1, :])
        # out = torch.cat([torch.max(out, dim=1)[0], torch.mean(out, dim=1)], dim=1)
        # out = self.relu(self.fc_3(out))
        # out = self.fc_4(out)

        # out = self.fc(out[:, -1, :])

        out = self.fc(out1)

        return out


class NewsLSTM_eye(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embedding):
        super(NewsLSTM_eye, self).__init__()
        self.emb = torch.eye(input_size)
        self.name = "NewsLSTM_eye"

        # 300 since vector is of size 300
        # self.emb = torch.eye(300)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers, dropout = 0.07, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.fc_1 = nn.Linear(hidden_size, 50)
        self.fc_2 = nn.Linear(50, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Look up the embedding
        x1 = self.emb[x[0]]
        # Set an initial hidden state
        h0 = torch.zeros(self.num_layers, len(x1), self.hidden_size)
        c0 = torch.zeros(self.num_layers, len(x1), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x1, (h0, c0))
        # Pass the output of the last time step to the classifier
        sliceTensor = x[1].long()-1

        sliceOut = out[:, sliceTensor, :]
        sliceOut2 = sliceOut[:, -1, :]

        out1 = self.relu(self.fc_1(sliceOut2))
        out2 = self.fc_2(out1)
        # out = torch.cat([torch.max(out, dim=1)[0], torch.mean(out, dim=1)], dim=1)
        # out = self.relu(self.fc_3(out))
        # out = self.fc_4(out)

        # out = self.fc(out[:, -1, :])

        return out2


class NewsLSTM_word2vec(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embedding):
        super(NewsLSTM_word2vec, self).__init__()
        self.emb = embedding
        self.relu = nn.ReLU()
        self.name = "NewsLSTM_word2vec"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers, dropout = 0.2, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Look up the embedding
        x1 = self.emb(x[0])
        # Set an initial hidden state
        h0 = torch.zeros(self.num_layers, len(x1), self.hidden_size)
        c0 = torch.zeros(self.num_layers, len(x1), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x1, (h0, c0))

        # Pass the output of the last time step to the classifier
        one = out[:, x[1].long()-1, :]
        out = self.relu(self.fc_1(one[:, -1, :]))
        out2 = self.fc_2(out)
        # out = torch.cat([torch.max(out, dim=1)[0], torch.mean(out, dim=1)], dim=1)
        # out = self.relu(self.fc_3(out))
        # out = self.fc_4(out)

        # out = self.fc(out[:, -1, :])

        return out2

# taken from https://stackoverflow.com/questions/53046583/how-to-create-a-torchtext-data-tabulardataset-directly-from-a-list-or-dict
# allows lists to be packaged into tabular dataset
class TabularDataset_From_List(data.Dataset):

    def __init__(self, input_list, format, fields, skip_header=False, **kwargs):
        make_example = {
            'json': torchtext.data.Example.fromJSON, 'dict': torchtext.data.Example.fromdict,
            'csv': torchtext.data.Example.fromCSV}[format.lower()]

        examples = [make_example(item, fields) for item in input_list]

        if make_example in (torchtext.data.Example.fromdict, torchtext.data.Example.fromJSON):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset_From_List, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):
        if path is None:
            path = cls.download(root)
        train_data = None if train is None else cls(
            train, **kwargs)
        val_data = None if validation is None else cls(
            validation, **kwargs)
        test_data = None if test is None else cls(
            test, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data

def tabular_dataSplit(glove_model):
    tokenizer = RegexpTokenizer(r'\w+')

    nltk.download('stopwords')

    stopList = stopwords.words('english')
    
    text_field = torchtext.data.Field(sequential=True,
                                    include_lengths=True,
                                    batch_first=True,
                                    use_vocab=True,
                                    tokenize=lambda x: [word.lower() for word in tokenizer.tokenize(x) if word.lower() not in stopList])
    # text_field = torchtext.data.Field(sequential=True,
    #                                 include_lengths=True,
    #                                 batch_first=True,
    #                                 use_vocab=True,
    #                                 tokenize=lambda x: torch.tensor([model_stoi[word].index for word in tokenizer.tokenize(x) if word in model_stoi]))
    label_field = torchtext.data.Field(sequential=False,
                                    use_vocab=False,
                                    is_target=True,      
                                    batch_first=True,
                                    preprocessing=lambda x: int(x == 1))
    base_field = torchtext.data.Field(sequential=False,
                                    use_vocab=True, 
                                    batch_first=True, 
                                    preprocessing=None)
    change_field = torchtext.data.Field(sequential=False,
                                    use_vocab=False,
                                    is_target=True,      
                                    batch_first=True,
                                    preprocessing=lambda x: int(x > 0))

    fields = {"title": ("title", text_field), 
    "label": ("label", label_field), 
    "description": ("description", text_field),
    "companyName": ("companyName", base_field), 
    "companySymbol": ("companySymbol", base_field), 
    "changePercent":("changePercent", change_field), 
    "label":("label", base_field)}

    # train, valid, test = dataSplitPkg.mainLoop(fileLoc = "finaldata_half.json")
    newsJson = loadJson(fileLoc = "json_data/finaldata_half.json")

    len_full = len(newsJson)

    # train = newsJson[:int(len_full*0.7)]
    # valid = newsJson[int(len_full*0.7):int(len_full*0.7)+int(len_full*0.15)]
    # test = newsJson[int(len_full*0.7)+int(len_full*0.15):]

    full_dataset = TabularDataset_From_List(input_list = newsJson, format = "dict", fields = fields)
    # train = TabularDataset_From_List(input_list = train, format = "dict", fields = fields)
    # valid = TabularDataset_From_List(input_list = valid, format = "dict", fields = fields)
    # test = TabularDataset_From_List(input_list = test, format = "dict", fields = fields)
    
    # old dataset split, works
    train, valid, test = full_dataset.split(split_ratio = [0.7, 0.15, 0.15])

    # len_full = len(full_dataset)
    # train = full_dataset[:int(len_full*0.7)]
    # valid = full_dataset[int(len_full*0.7):int(len_full*0.7)+int(len_full*0.15)]
    # test = full_dataset[int(len_full*0.7)+int(len_full*0.15):]

    # experimental split, doesn't work
    # train = torchtext.data.Dataset(train, fields)
    # valid = torchtext.data.Dataset(valid, fields)
    # test = torchtext.data.Dataset(test, fields)

    # ====================== word2vec translate ===========================
    # https://discuss.pytorch.org/t/aligning-torchtext-vocab-index-to-loaded-embedding-pre-trained-weights/20878
    modelLoc = "C:\\Temp\\GoogleNews-vectors-negative300.bin.gz"
    model = gensim.models.KeyedVectors.load_word2vec_format(
        modelLoc, binary=True, limit=50)

    # model.wv.save_word2vec_format("C:\\Temp\\word2vec.vec")

    vectors = Vectors(name="word2vec_10000000.vec", cache="C:\\Temp")

    # build vocab is needed to initialize vocab ca
    base_field.build_vocab(model.vocab)
    text_field.build_vocab(full_dataset, vectors=vectors)
    text_field.vocab.load_vectors(vectors=vectors)
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(text_field.vocab.vectors))

    # must have a initially built vocab
    # text_field.build_vocab(full_dataset)

    # text_field.build_vocab(vectors=vectors)
    # text_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

    return train, valid, test, model, text_field, embedding


def train_rnn_network(model, train_iter, valid_iter, num_epochs, learning_rate, batch_size, hidden_size):
    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc, train_prec, valid_prec = [], [], [], [], []
    best_model_acc = 0
    epochs = []
    errLog = []

    for epoch in range(num_epochs):
        model_path = "C:\\Temp\\best_models\\initial_models\\" + F'name{model.name}_bs{batch_size}_hs{hidden_size}_lr{learning_rate}_epoch{epoch+1}.pth'  

        for batch in train_iter:
            optimizer.zero_grad()
            pred = model(batch.title)
            loss = criterion(pred, batch.changePercent)
            loss.backward()
            optimizer.step()

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


def mainLoop():
    torch.manual_seed(2019)
    train, valid, test, model, text_field, embedding = tabular_dataSplit("glove_model")

    batch_size = 128
    hidden_size = 512

    train_iter = torchtext.data.BucketIterator(train,
                                           batch_size=batch_size,
                                           sort_key=lambda x: len(x.title), # to minimize padding
                                           sort_within_batch=True,        # sort within each batch
                                           repeat=False)                  # repeat the iterator for many epochs
    val_iter = torchtext.data.BucketIterator(valid,
                                           batch_size=batch_size,
                                           sort_key=lambda x: len(x.title), # to minimize padding
                                           sort_within_batch=True,        # sort within each batch
                                           repeat=False)                  # repeat the iterator for many epochs
    test_iter = torchtext.data.BucketIterator(test,
                                           batch_size=batch_size,
                                           sort_key=lambda x: len(x.title), # to minimize padding
                                           sort_within_batch=True,        # sort within each batch
                                           repeat=False)                  # repeat the iterator for many epochs
    
    input_size = embedding.num_embeddings
    
    model_LSTM_eye = NewsLSTM_eye(input_size, hidden_size, 2, 2, embedding)
    model_gru_eye = NewsGRU_eye(input_size, hidden_size, 2)

    # word2vec uses GoogleNews300 embedding
    model_gru_word2vec = NewsGRU_word2vec(300, hidden_size, 2, embedding)
    model_LSTM_word2vec = NewsLSTM_word2vec(300, hidden_size, 2, 2, embedding)

    # for any text value, vocab for the field must be built, otherwise torchtext throws error as it would be expecting integer

    # vectors = Vectors(name='xxx.vec', cache='./') https://github.com/pytorch/text/issues/201
    # text_field.build_vocab(train, valid, test, vectors=vectors)
    # get initial accuracy
    print("Get accuracy initialized.")
    print(get_accuracy(model_gru_word2vec, train_iter))
    print("Get accuracy complete.")

    train_rnn_network(model_gru_word2vec, train_iter, val_iter,
                    num_epochs=200, learning_rate=2e-04, batch_size=batch_size, hidden_size=hidden_size)


def test_model(model_path):
    train, valid, test, model, text_field, embedding = tabular_dataSplit("glove_model")
    batch_size = 128
    hidden_size = 256
    model_test = NewsLSTM_word2vec(300, hidden_size, 2, 2, embedding)
    model_test.load_state_dict(torch.load(model_path))

    val_iter = torchtext.data.BucketIterator(valid,
                                        batch_size=batch_size,
                                        sort_key=lambda x: len(x.headline), # to minimize padding
                                        sort_within_batch=True,        # sort within each batch
                                        repeat=False)                  # repeat the iterator for many epochs

    test_iter = torchtext.data.BucketIterator(test,
                                        batch_size=batch_size,
                                        sort_key=lambda x: len(x.headline), # to minimize padding
                                        sort_within_batch=True,        # sort within each batch
                                        repeat=False)                  # repeat the iterator for many epochs
    
    print(get_accuracy(model_test, val_iter))
    print(get_accuracy(model_test, test_iter))

# so that the training only runs when the python script is explicitly ran
# we don't want the training to begin when this script is imported
if __name__ == "__main__":
    mainLoop()

# test_model(model_path="C:\\Temp\\best_models\\initial_models\\nameNewsLSTM_word2vec_bs28_hs128_lr0.0002_epoch97.pth"