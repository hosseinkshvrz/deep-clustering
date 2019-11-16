import io
import numpy as np
from keras_preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
from gensim.models.wrappers import FastText


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(tkn) for tkn in tokens[1:]]
    return data


def get_embedding(data, document):
    tokens = text_to_word_sequence(document)
    tokens = [tkn for tkn in tokens if tkn != 'br']
    vector = np.zeros((256, 300,), dtype='float16')
    for i, tkn in enumerate(tokens):
        if i > 255:
            break
        try:
            # vc = data.get(tkn)
            vc = data[tkn]
            vector[i] = vc
        except KeyError:
            print(tkn)
    return vector


data = []
data_valid = []
label = []
label_valid = []
data_test = []
label_test = []

print('Here')

with open('data/aclImdb/train/data.txt') as data_file:
    for line in data_file:
        data.append(line.strip())
with open('data/aclImdb/train/label.txt') as target_file:
    for line in target_file:
        label.append(int(line.strip()))
with open('data/aclImdb/test/data.txt') as data_file:
    for line in data_file:
        data_test.append(line.strip())
with open('data/aclImdb/test/label.txt') as target_file:
    for line in target_file:
        label_test.append(int(line.strip()))

data_unsupervised = data[25000:]
label_unsupervised = label[25000:]

# it also does the shuffling
data, data_valid, label, label_valid = train_test_split(data[:25000],
                                                        label[:25000],
                                                        test_size=0.2,
                                                        random_state=1)

# model = load_vectors('data/wiki-news-300d-1M.vec')
model = FastText.load_fasttext_format('data/wiki.simple.bin')

print('vectors loaded')
x = np.zeros((len(data), 256, 300), dtype='float16')
for i, doc in enumerate(data):
    x[i] = get_embedding(model, doc)

np.save('data/fasttext/IMDB_train.npy', x)
print('train saved')


x_valid = np.zeros((len(data_valid), 256, 300), dtype='float16')
for i, doc in enumerate(data_valid):
    x_valid[i] = get_embedding(model, doc)

np.save('data/fasttext/IMDB_valid.npy', x_valid)
print('valid saved')


y = np.asarray(label)
np.save('data/fasttext/IMDB_trlabel.npy', y)
y_valid = np.asarray(label_valid)
np.save('data/fasttext/IMDB_valabel.npy', y_valid)
print('labels saved')

untagged = np.zeros((len(data_unsupervised), 256, 300), dtype='float16')
for i, doc in enumerate(data_unsupervised):
    untagged[i] = get_embedding(model, doc)

np.save('data/fasttext/IMDB_untagged.npy', untagged)

y_untagged = np.asarray(label_unsupervised)
np.save('data/fasttext/IMDB_unlabel.npy', y_untagged)

print('untagged saved')


x = np.zeros((len(data_test), 256, 300), dtype='float16')
for i, doc in enumerate(data_test):
    x[i] = get_embedding(model, doc)

np.save('data/fasttext/IMDB_test.npy', x)

y_test = np.asarray(label_test)
np.save('data/fasttext/IMDB_telabel.npy', y_test)

print(x.shape, ' test saved')

#

path = '/home/bsabeti/framework/data/'
file = 'IMDB_dauntagged_25000.npy'
data = np.load(path + file)
for i, d in enumerate(data):
    np.save(path + 'bert/' + 'unlabeled_' + str(i+1) + '.npy', d)

