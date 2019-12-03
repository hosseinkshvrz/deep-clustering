import argparse
import json
import os
import keras.backend as K
from keras import Input, Model, callbacks
from keras.engine import Layer, InputSpec
from keras.layers import Dense
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.utils import shuffle

import metrics

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_data():
    with open('/home/bsabeti/framework/data/reuters/other_files/data.txt') as file:
        data = file.readlines()

    data = [d.strip() for d in data]
    with open('/home/bsabeti/framework/data/reuters/other_files/labels.json') as json_file:
        labels = json.load(json_file)

    label = []
    for i in range(len(labels)):
        label.append(labels[str(i)+'.npy'])

    data, label = shuffle(data, label)
    valid, data = data[:10000], data[10000:]
    valabel, label = label[:10000], label[10000:]
    vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float64, sublinear_tf=True)
    x_sparse = vectorizer.fit_transform(data)
    x = np.asarray(x_sparse.todense())
    x_sparse = vectorizer.fit_transform(valid[:5000])
    x_valid = np.asarray(x_sparse.todense())
    y = np.asarray(label)
    y_valid = np.asarray(valabel[:5000])
    print('SST data shape ', x.shape)
    print('SST valid shape ', x_valid.shape)
    print("SST number of clusters: ", np.unique(y).size)
    return x, y, x_valid, y_valid


def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 init='glorot_uniform'):

        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, x_valid=None, y_valid=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining...')
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y, n_clusters):
                    self.x = x
                    self.y = y
                    self.n_clusters = n_clusters
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if epoch % 10 != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int((len(self.model.layers) - 1) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=self.n_clusters, n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    print('acc: {}'.format(metrics.inspect_clusters(self.y, y_pred, self.n_clusters)))

            cb.append(PrintACC(x_valid, y_valid, self.n_clusters))

        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, x_valid=None, y_valid=None, maxiter=20000, batch_size=200, tol=1e-3,
            update_interval=200, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                q_valid = self.model.predict(x_valid, verbose=0)

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                y_pred_valid = q_valid.argmax(1)
                if y is not None:
                    # acc = np.round(metrics.acc(y, y_pred), 5)
                    # nmi = np.round(metrics.nmi(y, y_pred), 5)
                    # ari = np.round(metrics.ari(y, y_pred), 5)
                    # loss = np.round(loss, 5)
                    # logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    # logwriter.writerow(logdict)
                    _, w = metrics.inspect_clusters(y, y_pred, self.n_clusters)
                    acc, _ = metrics.inspect_clusters(y_valid, y_pred_valid, self.n_clusters)
                    print('Iter {}, Acc: {} '.format(ite, acc), '; loss=', loss)

                p = self.target_distribution(q)

                # check stop criterion

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred_valid


if __name__ == "__main__":
    dataset = 'sst'
    batch_size = 200
    maxiter = 200000
    pretrain_epochs = 10
    update_interval = 200
    tol = 0.0001
    ae_weights = None
    save_dir = 'results10'
    n_clusters = 4

    # print(args)

    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x, y, x_valid, y_valid = load_data()
    # n_clusters = len(np.unique(y))

    # init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    # pretrain_optimizer = SGD(lr=1, momentum=0.9)

    # prepare the DEC model
    dec = DEC(dims=[x.shape[-1], 500, 10], n_clusters=n_clusters)

    if ae_weights is None:
        dec.pretrain(x=x, y=y, x_valid=x_valid, y_valid=y_valid, epochs=pretrain_epochs, batch_size=batch_size, save_dir=save_dir)
    else:
        dec.autoencoder.load_weights(ae_weights)

    dec.model.summary()
    dec.compile(loss='kld')
    y_pred = dec.fit(x, y=y, x_valid=x_valid, y_valid=y_valid, tol=tol, maxiter=maxiter, batch_size=batch_size,
                     update_interval=update_interval, save_dir=save_dir)
    print('acc:', metrics.inspect_clusters(y_valid, y_pred, n_clusters))
