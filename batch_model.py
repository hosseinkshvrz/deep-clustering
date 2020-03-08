# Deep Sentiment Clustering
import os
from time import time
import pickle
import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras import callbacks
from keras.utils import Sequence
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from auto_encoders import AutoEncoder
from metrics import inspect_clusters

path = os.path.dirname(os.path.abspath(__file__))


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


class DataGenerator(Sequence):
    def __init__(self, directory, data_files, doc_dims, batch_size=256):
        """
        :param directory: a string determining the directory containing the files
        :param data_files: a dictionary with two keys: labeled and unlabeled. Each maps to a list of file names.
        :param batch_size: the size of the batches
        """
        self.directory = directory
        self.labeled_files = data_files['labeled']
        self.unlabeled_files = data_files['unlabeled']
        self.doc_dims = doc_dims
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor((len(self.labeled_files) + len(self.unlabeled_files)) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        n_labeled = self.batch_size * len(self.labeled_files) // (len(self.labeled_files) + len(self.unlabeled_files))
        n_unlabeled = self.batch_size - n_labeled
        labeled_indexes = self.labeled_indexes[index * n_labeled:(index + 1) * n_labeled]
        unlabeled_indexes = self.unlabeled_indexes[index * n_unlabeled:(index + 1) * n_unlabeled]

        # Find list of IDs
        labeled_temp = [self.labeled_files[k] for k in labeled_indexes]
        unlabeled_temp = [self.unlabeled_files[k] for k in unlabeled_indexes]

        data, label = self.__data_generation(labeled_temp, unlabeled_temp)

        return data, label

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.labeled_indexes = np.arange(len(self.labeled_files))
        self.unlabeled_indexes = np.arange(len(self.unlabeled_files))
        np.random.shuffle(self.labeled_indexes)
        np.random.shuffle(self.unlabeled_indexes)

    def __data_generation(self, labeled, unlabeled):
        # data : (n_samples, *dim, n_channels)
        # Initialization
        files = labeled + unlabeled
        data = np.empty((len(files), *self.doc_dims), dtype='float16')
        label = np.empty((len(files), *self.doc_dims), dtype='float16')

        # Generate data
        for i, file_name in enumerate(files):
            data[i,] = np.load(self.directory + file_name)
            label[i,] = np.load(self.directory + file_name)

        data, label = shuffle(data, label)

        return data, label


class DSC(object):
    def __init__(self, directory, train_files, valid_file, labels, doc_dims, latent_dims, ae_type,
                 n_clusters, mask_file, mask_label=0.0, alpha=1.0, init='glorot_uniform'):
        super(DSC, self).__init__()
        self.directory = directory
        self.train_files = train_files
        self.valid_file = valid_file
        self.labels = labels
        self.doc_dims = doc_dims
        self.latent_dims = latent_dims
        self.ae_type = ae_type
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mask_label = mask_label
        with open(mask_file, 'rb') as file:
            self.mask_indexes = pickle.load(file)
        print('len mask file:', len(self.mask_indexes))
        self.mask_indexes = self.mask_indexes[:int(len(self.mask_indexes)*self.mask_label)]
        print('len mask file:', len(self.mask_indexes))
        cls = AutoEncoder(self.doc_dims, self.latent_dims, init=init)
        dataset_func = getattr(cls, ae_type)
        self.auto_encoder, self.encoder = dataset_func()
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, optimizer='adam', epochs=200,
                 batch_size=256, save_dir='results/'):
        print('Pre-training ...')

        training_generator = DataGenerator(self.directory, data_files=self.train_files,
                                           doc_dims=self.doc_dims, batch_size=batch_size)

        self.auto_encoder.summary()
        self.auto_encoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + 'pretrain_log.csv')
        cb = [csv_logger]

        class PrintACC(Callback):
            def __init__(self, v_file, n_clusters):
                self.v_file = v_file
                self.n_clusters = n_clusters
                super(PrintACC, self).__init__()

            def on_epoch_end(self, epoch, logs=None):
                if epoch % 10 != 0:
                    return
                x = np.load(self.v_file['data'])
                y = np.load(self.v_file['label'])
                feature_model = Model(self.model.input,
                                      self.model.get_layer(
                                          'encoder_%d' % (int((len(self.model.layers) - 1) / 2) - 1)).output)
                features = feature_model.predict(x)
                km = KMeans(n_clusters=self.n_clusters, n_init=20, n_jobs=4)
                y_pred = km.fit_predict(features)

                print('acc: {}'.format(inspect_clusters(y, y_pred, self.n_clusters)))

        cb.append(PrintACC(self.valid_file, self.n_clusters))

        # begin pre-training
        t0 = time()
        self.auto_encoder.fit_generator(generator=training_generator, epochs=epochs, callbacks=cb)
        print('Pre-training time: %ds' % round(time() - t0))
        self.auto_encoder.save(save_dir + 'ae_weights.h5')
        print('Pre-trained weights are saved to %sae_weights.h5' % save_dir)

    @staticmethod
    def target_distribution(q, y_true, w):
        weight = q ** 2 / q.sum(0)
        weight = (weight.T / weight.sum(1)).T
        # for index in range(len(weight)):
        #     if y_true[index] == 1:
        #         weight[index] = w[0]
        #     elif y_true[index] == 0:
        #         weight[index] = 1 - w[0]
        for index in range(len(weight)):
            if y_true[index] == 0:
                weight[index] = w[0]
            elif y_true[index] == 1:
                weight[index] = w[1]
            # elif y_true[index] == 2:
            #     weight[index] = w[2]
            # elif y_true[index] == 3:
            #     weight[index] = w[3]
        return weight

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def get_batch(self, labeled_files, unlabeled_files, labeled_indexes,
                  unlabeled_indexes, batch_index, batch_size):
        n_labeled = batch_size * len(labeled_files) // (len(labeled_files) + len(unlabeled_files))
        n_unlabeled = batch_size - n_labeled
        labeled_indexes = labeled_indexes[batch_index * n_labeled:(batch_index + 1) * n_labeled]
        unlabeled_indexes = unlabeled_indexes[batch_index * n_unlabeled:(batch_index + 1) * n_unlabeled]

        labeled_temp = [labeled_files[k] for k in labeled_indexes]
        unlabeled_temp = [unlabeled_files[k] for k in unlabeled_indexes]

        files = labeled_temp + unlabeled_temp
        data = np.empty((len(files), *self.doc_dims), dtype='float16')
        labels = np.empty(len(files), dtype='int32')

        # Generate data
        for i, file_name in enumerate(files):
            data[i,] = np.load(self.directory + file_name)
            labels[i] = self.labels[file_name]
            if file_name in self.mask_indexes:
                labels[i] = self.n_clusters

        data, labels = shuffle(data, labels)

        return data, labels

    def fit(self, max_iter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='results/'):
        save_embedding_interval = max_iter // 10
        print('Save embedding interval', save_embedding_interval)

        labeled_files = self.train_files['labeled']
        unlabeled_files = self.train_files['unlabeled']
        n_samples = len(labeled_files) + len(unlabeled_files)
        n_batches = int(np.floor(n_samples / batch_size))
        # Step 1: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        labeled_indexes = np.arange(len(labeled_files))
        unlabeled_indexes = np.arange(len(unlabeled_files))
        np.random.shuffle(labeled_indexes)
        np.random.shuffle(unlabeled_indexes)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        data = np.empty((0, *self.latent_dims), dtype='float16')
        for i in range(n_batches):
            x, _ = self.get_batch(labeled_files, unlabeled_files, labeled_indexes, unlabeled_indexes, i, batch_size)
            data = np.append(data, self.encoder.predict(x), axis=0)
        kmeans.fit(data)
        print('**** fit kmeans with all data ****')
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering

        best_acc = 0
        least_loss = np.inf
        w = np.zeros((self.n_clusters, self.n_clusters), dtype='int32')

        for ite in range(int(max_iter)):
            print('Epoch ', str(ite+1), '/', str(int(max_iter)))
            labeled_indexes = np.arange(len(labeled_files))
            unlabeled_indexes = np.arange(len(unlabeled_files))
            np.random.shuffle(labeled_indexes)
            np.random.shuffle(unlabeled_indexes)

            features = np.empty((0, *self.latent_dims), dtype='float16')
            labels = np.empty(0, dtype='int32')
            y_pred = np.empty(0, dtype='int32')
            y_true = np.empty(0, dtype='int32')

            epoch_loss = 0

            for i in range(n_batches):
                x, y = self.get_batch(labeled_files,
                                      unlabeled_files,
                                      labeled_indexes,
                                      unlabeled_indexes,
                                      i,
                                      batch_size)
                # print('batch loaded')

                if ite % save_embedding_interval == 0:
                    feature_model = Model(self.model.input,
                                          self.model.get_layer('encoder_%d' % (len(self.latent_dims) - 1)).output)
                    features = np.append(features, feature_model.predict(x), axis=0)
                    labels = np.append(labels, y, axis=0)

                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q, y, w)

                y_pred = np.append(y_pred, q.argmax(1), axis=0)
                y_true = np.append(y_true, y, axis=0)

                print('\n' + ' ' * 2 + str(i + 1) + '/' + str(n_batches) + ' ' +
                      '[' + '>' * (((i + 1) * 30 // n_batches) - 1) +
                      '.' * (30 - ((i + 1) * 30 // n_batches)) + ']', end=' ')

                if ite != 0:
                    loss = self.model.train_on_batch(x=x, y=p)
                    epoch_loss += loss
                    print('- loss =', loss, end='')

            print('\nstart inspecting clusters')
            if ite == 0:
                _, w = inspect_clusters(y_true, y_pred, self.n_clusters)

            print('start validating model')
            q_valid = self.model.predict(np.load(self.valid_file['data']), verbose=0)
            y_pred_valid = q_valid.argmax(1)
            acc, _ = inspect_clusters(np.load(self.valid_file['label']), y_pred_valid, self.n_clusters)
            print('acc =', acc, '; epoch loss= ', epoch_loss)
            if acc > best_acc:
                best_acc = acc
                print('saving model to:', save_dir + 'DEC_model_acc_' + str(ite) + '.h5')
                self.model.save(save_dir + 'DEC_model_acc_' + str(ite) + '.h5')
            if epoch_loss < least_loss:
                least_loss = epoch_loss
                print('saving model to:', save_dir + 'DEC_model_loss_' + str(ite) + '.h5')
                self.model.save(save_dir + 'DEC_model_loss_' + str(ite) + '.h5')

            if ite % save_embedding_interval == 0:
                np.save(save_dir + 'embedding_' + str(ite) + '.npy', features)
                np.save(save_dir + 'label_' + str(ite) + '.npy', labels)
                print('batch embedding saved')

        print('saving model to:', save_dir + 'DEC_model_final.h5')
        self.model.save(save_dir + 'DEC_model_final.h5')
