import os
from os.path import join
import numpy as np
from bert_serving.client import BertClient

path = os.path.dirname(os.path.abspath(__file__))


def load_imdb():
    data = []
    target = []
    data_test = []
    target_test = []
    with open(join(path, 'data/aclImdb/train/data.txt')) as data_file:
        for line in data_file:
            data.append(line.strip())
    with open(join(path, 'data/aclImdb/train/label.txt')) as target_file:
        for line in target_file:
            target.append(int(line.strip()))
    with open(join(path, 'data/aclImdb/test/data.txt')) as data_file:
        for line in data_file:
            data_test.append(line.strip())
    with open(join(path, 'data/aclImdb/test/label.txt')) as target_file:
        for line in target_file:
            target_test.append(int(line.strip()))

    # max_len_sent = max(data, key=len)
    # max_len = len(max_len_sent.split())
    # print(max_len)
    # longs = [d for d in data if len(d.split()) > 25]
    # print(len(longs))
    # bc = BertClient(ip='213.233.180.121')#, port='22')
    bc = BertClient()  # , port='22')
    # print('connection established')
    # vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float64, sublinear_tf=True)
    # x_sparse = vectorizer.fit_transform(data)
    # x = np.asarray(x_sparse.todense())
    vec = bc.encode(data[:25000])  # [9613, 32, 768]
    x = vec
    vec_test = bc.encode(data_test)
    x_test = vec_test
    y = np.asarray(target[:25000])
    y_test = np.asarray(target_test)
    print('IMDB data shape ', x.shape)
    print("IMDB number of clusters: ", np.unique(y).size)
    return x, y, x_test, y_test


def load_sst():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from bert_serving.client import BertClient
    data = []
    target = []
    with open(join(path, 'data/aclImdb/test/data.txt')) as data_file:
        for line in data_file:
            data.append(line.strip())
    with open(join(path, 'data/aclImdb/test/label.txt')) as target_file:
        for line in target_file:
            target.append(int(line.strip()))
    # max_len_sent = max(data, key=len)
    # max_len = len(max_len_sent.split())
    # print(max_len)
    # longs = [d for d in data if len(d.split()) > 25]
    # print(len(longs))
    # bc = BertClient(ip='213.233.180.121')#, port='22')
    bc = BertClient()#, port='22')
    # print('connection established')
    vec = bc.encode(data[:256])   # [9613, 32, 768]
    # vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float64, sublinear_tf=True)
    # x_sparse = vectorizer.fit_transform(data)
    # x = np.asarray(x_sparse.todense())
    x = vec
    y = np.asarray(target[:256])
    print('SST data shape ', x.shape)
    print("SST number of clusters: ", np.unique(y).size)
    return x, y


def load_data(dataset_name):
    if dataset_name == 'imdb':
        return load_imdb()
    elif dataset_name == 'sst':
        return load_sst()
    else:
        print('Not defined for loading', dataset_name)
        exit(0)
