import os
from os.path import join
import numpy as np
from bert_serving.client import BertClient
from sklearn.utils import shuffle

path = os.path.dirname(os.path.abspath(__file__))


class Dataset:
    def __init__(self, mode):
        self.data = []
        self.label = []
        self.data_test = []
        self.label_test = []
        self.mode = mode
        self.bc = BertClient()


class IMDB(Dataset):
    def __init__(self, *args, **kwargs):
        super(IMDB, self).__init__(*args, **kwargs)

    def get_data(self):
        with open(join(path, 'data/aclImdb/train/data.txt')) as data_file:
            for line in data_file:
                self.data.append(line.strip())
        with open(join(path, 'data/aclImdb/train/label.txt')) as target_file:
            for line in target_file:
                self.label.append(int(line.strip()))

        if self.mode == 'supervised':
            self.data = self.data[:25000]
            self.label = self.label[:25000]

        x = self.bc.encode(self.data)
        y = np.asarray(self.label)
        print('IMDB data shape ', x.shape)
        print("IMDB number of clusters: ", np.unique(y).size)
        # original data in IMDB dataset is in order
        x, y = shuffle(x, y)
        return x, y

    def get_test_data(self):
        with open(join(path, 'data/aclImdb/test/data.txt')) as data_file:
            for line in data_file:
                self.data_test.append(line.strip())
        with open(join(path, 'data/aclImdb/test/label.txt')) as target_file:
            for line in target_file:
                self.label_test.append(int(line.strip()))

        x = self.bc.encode(self.data_test)
        y = np.asarray(self.label_test)
        print('IMDB test data shape ', x.shape)
        print("IMDB number of clusters: ", np.unique(y).size)
        # original data in IMDB dataset is in order
        x, y = shuffle(x, y)
        return x, y


class SST(Dataset):
    def __init__(self, *args, **kwargs):
        super(SST, self).__init__(*args, **kwargs)

    def get_data(self):
        with open(join(path, 'data/sst/train/data.txt')) as data_file:
            for line in data_file:
                self.data.append(line.strip())
        with open(join(path, 'data/sst/train/label.txt')) as target_file:
            for line in target_file:
                self.label.append(int(line.strip()))

        if self.mode == 'semi-supervised':
            raise ValueError('SST does not have untagged data.')

        x = self.bc.encode(self.data)
        y = np.asarray(self.label)
        print('SST data shape ', x.shape)
        print("SST number of clusters: ", np.unique(y).size)
        # original data in IMDB dataset is in order
        x, y = shuffle(x, y)
        return x, y

    def get_test_data(self):
        with open(join(path, 'data/sst/test/data.txt')) as data_file:
            for line in data_file:
                self.data_test.append(line.strip())
        with open(join(path, 'data/sst/test/label.txt')) as target_file:
            for line in target_file:
                self.label_test.append(int(line.strip()))

        x = self.bc.encode(self.data_test)
        y = np.asarray(self.label_test)
        print('SST test data shape ', x.shape)
        print("SST number of clusters: ", np.unique(y).size)
        # original data in IMDB dataset is in order
        x, y = shuffle(x, y)
        return x, y
