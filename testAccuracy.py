import torch
import torch.nn as nn
import initial_model
import torchtext
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from torchtext.vocab import Vectors
import gensim
import torch.nn.functional as F
import logging
import tensorflow as tf


def test_model(model_path, hidden_size):
    train, valid, test, model, text_field, embedding = initial_model.tabular_dataSplit("glove_model")

    model_test = initial_model.NewsLSTM_word2vec(300, hidden_size, 2, 2, embedding)
    model_test.load_state_dict(torch.load(model_path))

    return model_test


def load_demo_iter(input_json, newsJson):
    tokenizer = RegexpTokenizer(r'\w+')
    stopList = stopwords.words('english')
    text_field = torchtext.data.Field(sequential=True,
                                    include_lengths=True,
                                    batch_first=True,
                                    use_vocab=True,
                                    tokenize=lambda x: [word.lower() for word in tokenizer.tokenize(x) if word.lower() not in stopList])
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

    modelLoc = "C:\\Temp\\GoogleNews-vectors-negative300.bin.gz"
    model = gensim.models.KeyedVectors.load_word2vec_format(
        modelLoc, binary=True, limit=50)

    # model.wv.save_word2vec_format("C:\\Temp\\word2vec.vec")
    demo_dataset = initial_model.TabularDataset_From_List(input_list = input_json, format = "dict", fields = fields)
    full_dataset = initial_model.TabularDataset_From_List(input_list = newsJson, format = "dict", fields = fields)

    # build vocab is needed to initialize vocab ca
    base_field.build_vocab(model.vocab)
    vectors = Vectors(name="word2vec_10000000.vec", cache="C:\\Temp")
    text_field.build_vocab(full_dataset, vectors=vectors)
    text_field.vocab.load_vectors(vectors=vectors)

    test_iter = torchtext.data.BucketIterator(
        demo_dataset,
        batch_size=128,
        sort_key=lambda x: len(x.title),
        sort_within_batch=True,
        repeat=False)          
        
    return test_iter


def get_demo_result(model, demo_iter):
    output_list = []
    for batch in demo_iter:
        output = model(batch.title)

        # apply some sort of softmax
        output_list.append(output)

    return output_list[0]


def get_accuracy(model, data_loader):
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

    return correct/total, ha


def mainLoop():
    # logging.getLogger('smart_open').propagate = False
    # logging.getLogger('smart_open').setLevel(logging.CRITICAL)
    print("loading model...")
    model = test_model(
        model_path = "C:\\Temp\\best_models\\initial_models\\nameNewsLSTM_word2vec_bs256_hs248_lr0.00035_epoch45.pth",
        hidden_size = 248)

    newsJson = initial_model.loadJson(fileLoc="C:\\Temp\\finalData_big_balanced_keyTitle.json")
    totalJson = initial_model.loadJson(fileLoc="json_data\\finaldata_half.json")
    print("loading complete")
    
    test_iter = load_demo_iter(newsJson, totalJson)

    acc, prec = get_accuracy(model, test_iter)

    print("acc:", acc)
    print("prec:", prec)


mainLoop()
