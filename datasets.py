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
        print('Here')
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

        filepath = path + '/data/IMDB_valabel.npy'
        exists = os.path.isfile(filepath)
        # the following procedure is due to the randomness of the split function
        if not exists:
            x = self.bc.encode(self.data)
            np.save(path + '/data/IMDB_train.npy', x)
            y = np.asarray(self.label)
            np.save(path + '/data/IMDB_trlabel.npy', y)
            x_valid = self.bc.encode(self.data_valid)
            np.save(path + '/data/IMDB_valid.npy', x_valid)
            y_valid = np.asarray(self.label_valid)
            np.save(path + '/data/IMDB_valabel.npy', y_valid)
        else:
            x = np.load(path + '/data/IMDB_train.npy')
            x = x.astype('float16')
            y = np.load(path + '/data/IMDB_trlabel.npy')
            x_valid = np.load(path + '/data/IMDB_valid.npy')
            x_valid = x_valid.astype('float16')
            y_valid = np.load(path + '/data/IMDB_valabel.npy')

        print('tagged loaded')

        if self.mode == 'semi-supervised':
            # filepath = path + '/data/IMDB_dauntagged.npy'
            # exists = os.path.isfile(filepath)
            # if not exists:
            #     data_unsupervised = self.bc.encode(data_unsupervised)
            #     np.save(path + '/data/IMDB_dauntagged.npy', data_unsupervised)
            #     label_unsupervised = np.asarray(label_unsupervised)
            #     np.save(path + '/data/IMDB_launtagged.npy', label_unsupervised)
            # else:
            #     data_unsupervised = np.load(path + '/data/IMDB_dauntagged.npy')
            #     label_unsupervised = np.load(path + 'data/IMDB_launtagged.npy')
            data_unsupervised = np.load(path + '/data/IMDB_dauntagged_5000.npy')
            data_unsupervised = data_unsupervised.astype('float16')
            x = np.append(x, data_unsupervised, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_10000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_15000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_20000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_25000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_30000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_35000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_40000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_45000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')
            temp = np.load(path + '/data/IMDB_dauntagged_50000.npy')
            temp = temp.astype('float16')
            # data_unsupervised = np.append(data_unsupervised, temp, axis=0)
            x = np.append(x, temp, axis=0)
            print('chunk loaded')

            label_unsupervised = np.asarray(label_unsupervised)

            # x = np.append(x, data_unsupervised, axis=0)
            y = np.append(y, label_unsupervised, axis=0)

            print('untagged loaded')

        print('IMDB data shape ', x.shape)
        print('IMDB validation data shape ', x_valid.shape)
        print("IMDB number of clusters: ", np.unique(y).size)
        # original data in IMDB dataset is in order
        # no need to shuffle the validation data since it is shuffled in the val selection phase
        x, y = shuffle(x, y)
        print('shuffled')
        return x, y, x_valid, y_valid

    def get_test_data(self):
        print('In the beginning of get test data')
        with open(join(path, 'data/aclImdb/test/data.txt')) as data_file:
            for line in data_file:
                self.data_test.append(line.strip())
        with open(join(path, 'data/aclImdb/test/label.txt')) as target_file:
            for line in target_file:
                self.label_test.append(int(line.strip()))
        print('before loading')
        filepath = path + '/data/IMDB_test.npy'
        exists = os.path.isfile(filepath)
        if not exists:
            x = self.bc.encode(self.data_test)
            np.save(filepath, x)
        else:
            x = np.load(filepath)
        print('loaded!!!!')
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
        filepath = path + '/data/SST_valabel_' + self.mode + '.npy'
        exists = os.path.isfile(filepath)
        # the following procedure is due to the randomness of the split function
        if not exists:
            x = self.bc.encode(self.data)
            np.save(path + '/data/SST_train_' + self.mode + '.npy', x)
            y = np.asarray(self.label)
            np.save(path + '/data/SST_trlabel_' + self.mode + '.npy', y)
            x_valid = self.bc.encode(self.data_valid)
            np.save(path + '/data/SST_valid_' + self.mode + '.npy', x_valid)
            y_valid = np.asarray(self.label_valid)
            np.save(path + '/data/SST_valabel_' + self.mode + '.npy', y_valid)
        else:
            x = np.load(path + '/data/SST_train_' + self.mode + '.npy')
            y = np.load(path + '/data/SST_trlabel_' + self.mode + '.npy')
            x_valid = np.load(path + '/data/SST_valid_' + self.mode + '.npy')
            y_valid = np.load(path + '/data/SST_valabel_' + self.mode + '.npy')

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

        filepath = path + '/data/SST_test.npy'
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
