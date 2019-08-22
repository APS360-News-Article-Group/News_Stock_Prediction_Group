import torch
import torch.nn as nn
import initial_model_moreTrain
import torchtext
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from torchtext.vocab import Vectors
import gensim
import torch.nn.functional as F
import logging
import initial_model


class NewsLSTM_word2vec(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embedding):
        super(NewsLSTM_word2vec, self).__init__()
        self.emb = embedding
        self.name = "NewsLSTM_word2vec"

        # 300 since vector is of size 300
        # self.emb = torch.eye(300)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers, dropout = 0.2, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 50)
        self.fc_2 = nn.Linear(50, num_classes)
        self.relu = nn.ReLU()

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


def test_model(model_path, hidden_size):
    train, valid, test, model, text_field, embedding = initial_model_moreTrain.tabular_dataSplit("glove_model")

    model_test = initial_model_moreTrain.NewsLSTM_word2vec(300, hidden_size, 2, 2, embedding)
    model_test.load_state_dict(torch.load(model_path))

    return model_test


def load_test_iter(input_json, newsJson):
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
    demo_dataset = initial_model_moreTrain.TabularDataset_From_List(input_list = input_json, format = "dict", fields = fields)
    full_dataset = initial_model_moreTrain.TabularDataset_From_List(input_list = newsJson, format = "dict", fields = fields)

    # build vocab is needed to initialize vocab ca
    base_field.build_vocab(model.vocab)
    vectors = Vectors(name="word2vec_10000000.vec", cache="C:\\Temp")
    text_field.build_vocab(full_dataset, vectors=vectors)
    text_field.vocab.load_vectors(vectors=vectors)

    test_iter = torchtext.data.BucketIterator(
        demo_dataset,
        batch_size=64,
        sort_key=lambda x: len(x.title),
        sort_within_batch=True,
        repeat=False)          
        
    return test_iter


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
    demo_dataset = initial_model_moreTrain.TabularDataset_From_List(input_list = input_json, format = "dict", fields = fields)
    full_dataset = initial_model_moreTrain.TabularDataset_From_List(input_list = newsJson, format = "dict", fields = fields)

    # build vocab is needed to initialize vocab ca
    base_field.build_vocab(model.vocab)
    vectors = Vectors(name="word2vec_10000000.vec", cache="C:\\Temp")
    text_field.build_vocab(full_dataset, vectors=vectors)
    text_field.vocab.load_vectors(vectors=vectors)

    demo_iter = torchtext.data.BucketIterator(
        demo_dataset,
        batch_size=1,
        sort_key=lambda x: len(x.title),
        sort_within_batch=True,
        repeat=False)          
        
    return demo_iter


def get_demo_result(model, demo_iter):
    # testing, thus train mode is false
    model.train(False)
    output_list = []
    for batch in demo_iter:
        output = model(batch.title)

        # apply some sort of softmax
        output_list.append(output)

    return output_list[0]


def mainLoop():
    # logging.getLogger('smart_open').propagate = False
    # logging.getLogger('smart_open').setLevel(logging.CRITICAL)
    print("loading model...")
    model = test_model(
        model_path = "C:\\Temp\\best_models\\initial_models\\nameNewsLSTM_word2vec_bs32_hs256_lr0.0002_epoch69.pth",
        hidden_size = 256)

    newsJson = initial_model_moreTrain.loadJson(fileLoc = "json_data/finaldata_half.json")
    print("loading complete")

    while True:
        input_title = input("Please input a demo news title: ")

        input_json = [{
            "title": input_title,
            "label": 0, 
            "description": "some_description",
            "companyName": "some_companyName", 
            "companySymbol": "some_companySymbol",
            "changePercent": 0}]
        
        demo_iter = load_demo_iter(input_json, newsJson)

        result = get_demo_result(model, demo_iter)

        prediction = F.softmax(result, dim=1)

        print(prediction.data)


def testLoop():
    # logging.getLogger('smart_open').propagate = False
    # logging.getLogger('smart_open').setLevel(logging.CRITICAL)
    print("loading model...")
    model = test_model(
        model_path = "C:\\Temp\\best_models\\initial_models\\nameNewsLSTM_word2vec_bs32_hs256_lr0.0002_epoch69.pth",
        hidden_size = 256)

    newsJson = initial_model_moreTrain.loadJson(fileLoc = "json_data/finaldata_half.json")
    testJson = initial_model_moreTrain.loadJson(fileLoc = "json_data/gnews_data_Jun17_Jun28_with_stock.json")
    # testJson = initial_model_moreTrain.loadJson(fileLoc = "json_data/gnews_data_July1_July10_with_stock.json")
    print("loading complete")
    test_iter = load_test_iter(testJson, newsJson)

    # prediction, precision = initial_model_moreTrain.get_accuracy(model, test_iter)
    prediction, precision = initial_model.get_accuracy(model, test_iter)

    print("prediction acc:", prediction)
    print("precision acc:", precision)

    print("done")


if __name__ == "__main__":
    testLoop()
    # mainLoop()
