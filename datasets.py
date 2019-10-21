import os
from os.path import join
import numpy as np
from bert_serving.client import BertClient
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

path = os.path.dirname(os.path.abspath(__file__))


class Dataset:
    def __init__(self, mode):
        self.data = []
        self.label = []
        self.data_valid = []
        self.label_valid = []
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

        data_unsupervised = self.data[25000:]
        label_unsupervised = self.label[25000:]
        # it also does the shuffling
        self.data, self.data_valid, self.label, self.label_valid = train_test_split(self.data[:25000],
                                                                                    self.label[:25000],
                                                                                    test_size=0.2,
                                                                                    random_state=1)

        if self.mode == 'semi-supervised':
            self.data = self.data + data_unsupervised
            self.label = self.label + label_unsupervised

        filepath = path + 'data/IMDB_train_' + self.mode + '.npy'
        exists = os.path.isfile(filepath)
        if not exists:
            x = self.bc.encode(self.data)
            np.save(filepath, x)
        else:
            x = np.load(filepath)

        y = np.asarray(self.label)

        filepath = path + 'data/IMDB_valid_' + self.mode + '.npy'
        exists = os.path.isfile(filepath)
        if not exists:
            x_valid = self.bc.encode(self.data_valid)
            np.save(filepath, x_valid)
        else:
            x_valid = np.load(filepath)

        y_valid = np.asarray(self.label_valid)

        print('IMDB data shape ', x.shape)
        print('IMDB validation data shape ', x_valid.shape)
        print("IMDB number of clusters: ", np.unique(y).size)
        # original data in IMDB dataset is in order
        # no need to shuffle the validation data since it is shuffled in the val selection phase
        x, y = shuffle(x, y)
        return x, y, x_valid, y_valid

    def get_test_data(self):
        with open(join(path, 'data/aclImdb/test/data.txt')) as data_file:
            for line in data_file:
                self.data_test.append(line.strip())
        with open(join(path, 'data/aclImdb/test/label.txt')) as target_file:
            for line in target_file:
                self.label_test.append(int(line.strip()))

        filepath = path + 'data/IMDB_test.npy'
        exists = os.path.isfile(filepath)
        if not exists:
            x = self.bc.encode(self.data_test)
            np.save(filepath, x)
        else:
            x = np.load(filepath)

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

        self.data, self.data_valid, self.label, self.label_valid = train_test_split(self.data,
                                                                                    self.label,
                                                                                    test_size=0.2,
                                                                                    random_state=1)
        filepath = path + 'data/SST_train.npy'
        exists = os.path.isfile(filepath)
        if not exists:
            x = self.bc.encode(self.data)
            np.save(filepath, x)
        else:
            x = np.load(filepath)

        y = np.asarray(self.label)

        filepath = path + 'data/SST_valid.npy'
        exists = os.path.isfile(filepath)
        if not exists:
            x_valid = self.bc.encode(self.data_valid)
            np.save(filepath, x_valid)
        else:
            x_valid = np.load(filepath)

        y_valid = np.asarray(self.label_valid)

        print('SST data shape ', x.shape)
        print('SST validation data shape ', x_valid.shape)
        print("SST number of clusters: ", np.unique(y).size)
        # SST does not need shuffling
        return x, y, x_valid, y_valid

    def get_test_data(self):
        with open(join(path, 'data/sst/test/data.txt')) as data_file:
            for line in data_file:
                self.data_test.append(line.strip())
        with open(join(path, 'data/sst/test/label.txt')) as target_file:
            for line in target_file:
                self.label_test.append(int(line.strip()))

        filepath = path + 'data/SST_test.npy'
        exists = os.path.isfile(filepath)
        if not exists:
            x = self.bc.encode(self.data_test)
            np.save(filepath, x)
        else:
            x = np.load(filepath)

        y = np.asarray(self.label_test)

        print('SST test data shape ', x.shape)
        print("SST number of clusters: ", np.unique(y).size)
        return x, y
