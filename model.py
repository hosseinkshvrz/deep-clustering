# Deep Sentiment Clustering

from time import time
import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras import callbacks
from sklearn.cluster import KMeans

from auto_encoders import AutoEncoder
from metrics import inspect_clusters


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


class DSC(object):
    def __init__(self, doc_dims, latent_dims, ae_type, n_clusters, alpha=1.0, init='glorot_uniform'):
        super(DSC, self).__init__()
        self.doc_dims = doc_dims
        self.latent_dims = latent_dims
        self.ae_type = ae_type
        self.n_clusters = n_clusters
        self.alpha = alpha
        cls = AutoEncoder(self.doc_dims, self.latent_dims, init=init)
        dataset_func = getattr(cls, ae_type)
        self.auto_encoder, self.encoder = dataset_func()
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def maybe_new_fit(self, x, y=None, x_valid=None, y_valid=None, max_iter=2e4, batch_size=256,
                      update_interval=140, save_dir='results/'):
        print('Update interval', update_interval)
        get_acc_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Get accuracy interval', get_acc_interval)

        # Step 1: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        class UpdateParams(Callback):
            def __init__(self, x, y, x_valid, y_valid, n_clusters):
                self.x = x
                self.y = y
                self.x_valid = x_valid
                self.y_valid = y_valid
                self.n_clusters = n_clusters
                self.filepath = save_dir + 'model.h5'
                super(UpdateParams, self).__init__()

            @staticmethod
            def target_distribution(q, y_true, w):
                weight = q ** 2 / q.sum(0)
                for index in range(len(weight)):
                    if y_true[index] == 1:
                        weight[index] = w[0]
                    elif y_true[index] == 0:
                        weight[index] = 1 - w[0]
                weight = (weight.T / weight.sum(1)).T
                return weight

            def on_epoch_end(self, epoch, logs=None):
                if epoch % update_interval != 0:
                    return
                # A callback has access to its associated model through the class property self.model
                q = self.model.predict(self.x, verbose=0)
                q_valid = self.model.predict(self.x_valid, verbose=0)
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                y_pred_valid = q_valid.argmax(1)
                _, w = inspect_clusters(y, y_pred, self.n_clusters)
                acc, _ = inspect_clusters(y_valid, y_pred_valid, self.n_clusters)
                print('Iter {}, Acc: {} '.format(epoch, acc))
                self.p = self.target_distribution(q, y, w)
                monitor_op = np.greater
                best = -np.Inf
                if monitor_op(acc, best):
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s'
                          % (epoch + 1, 'acc', best, acc, self.filepath))
                    best = acc
                    self.model.save(self.filepath, overwrite=True)
                else:
                    print('\nEpoch %05d: %s did not improve from %0.5f'
                          % (epoch + 1, 'acc', best))

        update_params = UpdateParams(x, y, x_valid, y_valid, self.n_clusters)
        cb = [update_params]
        self.model.fit(x=x, y=update_params.p, batch_size=batch_size, epochs=max_iter, callbacks=cb)

    def pretrain(self, x, y=None, x_valid=None, y_valid=None, optimizer='adam', epochs=200,
                 batch_size=256, save_dir='results/'):
        print('Pre-training ...')
        self.auto_encoder.summary()
        self.auto_encoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(Callback):
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

                    print('acc: {}'.format(inspect_clusters(self.y, y_pred, self.n_clusters)))

            cb.append(PrintACC(x_valid, y_valid, self.n_clusters))

        # begin pre-training
        t0 = time()
        self.auto_encoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pre-training time: %ds' % round(time() - t0))
        self.auto_encoder.save(save_dir + '/ae_weights.h5')
        print('Pre-trained weights are saved to %s/ae_weights.h5' % save_dir)

    @staticmethod
    def target_distribution(q, y_true, w):
        weight = q ** 2 / q.sum(0)
        for index in range(len(weight)):
            if y_true[index] == 1:
                weight[index] = w[0]
            elif y_true[index] == 0:
                weight[index] = 1 - w[0]
        weight = (weight.T / weight.sum(1)).T
        return weight

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, x_valid=None, y_valid=None, max_iter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='results/'):

        print('Update interval', update_interval)
        get_acc_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Get accuracy interval', get_acc_interval)

        # Step 1: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        best_acc = 0
        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(max_iter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                q_valid = self.model.predict(x_valid, verbose=0)

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                y_pred_valid = q_valid.argmax(1)
                if y is not None:
                    _, w = inspect_clusters(y, y_pred, self.n_clusters)
                    acc, _ = inspect_clusters(y_valid, y_pred_valid, self.n_clusters)
                    print('Iter {}, Acc: {} '.format(ite, acc), '; loss=', loss)
                    if acc > best_acc:
                        best_acc = acc
                        print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                        self.model.save(save_dir + '/DEC_model_' + str(ite) + '.h5')

                p = self.target_distribution(q, y, w)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        # save the trained model
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save(save_dir + '/DEC_model_final.h5')
