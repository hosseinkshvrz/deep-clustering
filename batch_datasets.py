import os
import pickle
from os import listdir
from os.path import isfile, join
import json
import numpy as np
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


class IMDB(Dataset):
    def __init__(self, *args, **kwargs):
        super(IMDB, self).__init__(*args, **kwargs)

    def get_data(self):
        print('Here')
        directory = '/home/bsabeti/framework/data/bert/'
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        labeled = [f for f in files if f.startswith('labeled_')]
        unlabeled = list()
        if self.mode == 'semi-supervised':
            unlabeled = [f for f in files if f.startswith('unlabeled_')]

        print('labeled size: ', len(labeled))
        print('unlabeled size: ', len(unlabeled))

        train_files = {'labeled': labeled, 'unlabeled': unlabeled}
        valid_file = {'data': '/home/bsabeti/framework/data/IMDB_valid.npy',
                      'label': '/home/bsabeti/framework/data/IMDB_valabel.npy'}
        with open('/home/bsabeti/framework/data/labels.json') as json_file:
            labels = json.load(json_file)
        doc_dims = np.load(directory + labeled[0]).shape
        print('dims: ', doc_dims)

        return directory, train_files, valid_file, labels, doc_dims

    def get_test_data(self):
        print('In the beginning of get test data')
        with open(join(path, 'data/aclImdb/test/data.txt')) as data_file:
            for line in data_file:
                self.data_test.append(line.strip())
        with open(join(path, 'data/aclImdb/test/label.txt')) as target_file:
            for line in target_file:
                self.label_test.append(int(line.strip()))
        print('before loading')
        filepath = path + '/data/fasttext/IMDB_test.npy'
        exists = os.path.isfile(filepath)
        if not exists:
            return None
            # x = self.bc.encode(self.data_test)
            # np.save(filepath, x)
        else:
            x = np.load(filepath)
        print('loaded!!!!')
        y = np.asarray(self.label_test)

        print('IMDB test data shape ', x.shape)
        print("IMDB number of clusters: ", np.unique(y).size)
        # original data in IMDB dataset is in order
        x, y = shuffle(x, y)
        return x, y


class Reuters(Dataset):
    def __init__(self, *args, **kwargs):
        super(Reuters, self).__init__(*args, **kwargs)

    def get_data(self):
        print('Here')
        directory = '/home/bsabeti/framework/data/reuters/'
        mask_file = '/home/bsabeti/framework/data/reuters/other_files/reuters_mask.txt'
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        print('size: ', len(files))
        train_files = {'labeled': files, 'unlabeled': []}

        if not os.path.exists(mask_file):
            np.random.shuffle(files)
            with open(mask_file, 'wb') as file:
                pickle.dump(files, file)

        valid_file = {'data': '/home/bsabeti/framework/data/reuters_valid.npy',
                      'label': '/home/bsabeti/framework/data/reuters_valabel.npy'}
        with open('/home/bsabeti/framework/data/reuters/other_files/labels.json') as json_file:
            labels = json.load(json_file)
        doc_dims = np.load(directory + files[0]).shape
        print('dims: ', doc_dims)

        return directory, train_files, valid_file, labels, doc_dims, mask_file
