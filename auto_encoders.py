from keras import Input, Model
from keras.layers import LSTM, RepeatVector, Dense, Flatten, Reshape, GRU
import numpy as np


class AutoEncoder:
    """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
    """

    def __init__(self, doc_dims, latent_dims, act='relu', init='glorot_uniform'):
        self.doc_dims = doc_dims
        self.act = act
        self.init = init
        self.latent_dims = latent_dims
        self.n_stacks = len(self.latent_dims)
        self.input_layer = Input(shape=self.doc_dims, name='input')

    def lstm_ae(self):
        h = self.input_layer

        # internal layers in encoder
        for i in range(self.n_stacks - 1):
            h = LSTM(self.latent_dims[i], name='encoder_%d' % i)(h)

        # hidden layer
        h = LSTM(self.latent_dims[-1], name='encoder_%d' % (self.n_stacks - 1))(h)  # hidden layer, features are extracted from here

        y = h

        # internal layers in decoder
        for i in range(self.n_stacks-1, 0, -1):
            y = LSTM(self.latent_dims[i], name='decoder_%d' % i)(y)

        # output
        decoded = RepeatVector(self.doc_dims[0])(y)
        y = LSTM(self.doc_dims[1], name='decoder_0', return_sequences=True)(decoded)

        return Model(inputs=self.input_layer, outputs=y, name='AE'), Model(inputs=self.input_layer, outputs=h, name='encoder')

    def gru_ae(self):
        h = self.input_layer

        # internal layers in encoder
        for i in range(self.n_stacks - 1):
            h = GRU(self.latent_dims[i], name='encoder_%d' % i)(h)

        # hidden layer
        h = GRU(self.latent_dims[-1], name='encoder_%d' % (self.n_stacks - 1))(h)  # hidden layer, features are extracted from here

        y = h

        # internal layers in decoder
        for i in range(self.n_stacks-1, 0, -1):
            y = GRU(self.latent_dims[i], name='decoder_%d' % i)(y)

        # output
        decoded = RepeatVector(self.doc_dims[0])(y)
        y = GRU(self.doc_dims[1], name='decoder_0', return_sequences=True)(decoded)

        return Model(inputs=self.input_layer, outputs=y, name='AE'), Model(inputs=self.input_layer, outputs=h, name='encoder')

    def dense_ae(self):
        h = self.input_layer

        h = Flatten()(h)

        # internal layers in encoder
        for i in range(self.n_stacks - 1):
            h = Dense(self.latent_dims[i], activation=self.act, kernel_initializer=self.init, name='encoder_%d' % i)(h)

        # hidden layer
        h = Dense(self.latent_dims[-1], kernel_initializer=self.init, name='encoder_%d' % (self.n_stacks - 1))(h)  # hidden layer, features are extracted from here

        y = h

        # internal layers in decoder
        for i in range(self.n_stacks-1, 0, -1):
            y = Dense(self.latent_dims[i], activation=self.act, kernel_initializer=self.init, name='decoder_%d' % i)(y)

        # output
        y = Dense(np.prod(self.doc_dims), kernel_initializer=self.init, name='decoder_0')(y)

        y = Reshape(self.doc_dims)

        return Model(inputs=self.input_layer, outputs=y, name='AE'), Model(inputs=self.input_layer, outputs=h, name='encoder')
