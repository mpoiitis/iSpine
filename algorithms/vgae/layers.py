from tensorflow.keras.layers import Layer
import tensorflow as tf


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalEncoder(Layer):

    def __init__(self, input_dim, output_dim, hidden_dim, num_features_nonzero, activation, dropout, is_sparse_inputs=True,  name="variational_encoder", **kwargs):
        super(VariationalEncoder, self).__init__(name=name, **kwargs)
        self.proj = GraphConvolution(input_dim=input_dim, output_dim=hidden_dim, num_features_nonzero=num_features_nonzero,
                                 activation=activation, dropout=dropout, is_sparse_inputs=True)
        self.mean = GraphConvolution(input_dim=hidden_dim, output_dim=output_dim, num_features_nonzero=num_features_nonzero,
                                activation=lambda x: x, dropout=dropout)
        self.log_var = GraphConvolution(input_dim=hidden_dim, output_dim=output_dim, num_features_nonzero=num_features_nonzero,
                                activation=lambda x: x, dropout=dropout)
        self.sampling = Sampling()

    def call(self, inputs, training):
        x, support = inputs
        h = self.proj((x, support), training)
        z_mean = self.mean((h, support), training)
        z_log_var = self.log_var((h, support), training)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Encoder(Layer):

    def __init__(self, input_dim, output_dim, num_features_nonzero, activation, dropout, is_sparse_inputs=True, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.proj = GraphConvolution(input_dim=input_dim, output_dim=output_dim, num_features_nonzero=num_features_nonzero,
                                 activation=activation, dropout=dropout, is_sparse_inputs=True)

    def call(self, inputs, training):
        z = self.proj(inputs, training)
        return z


class Decoder(Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dropout = dropout
        self.act = act

    def call(self, inputs):
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))


class GraphConvolution(Layer):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero

        self.weights_ = []
        for i in range(1):
            w = self.add_variable('weight' + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_variable('bias', [output_dim])



    def call(self, inputs, training=None):
        x, support_ = inputs

        # dropout
        if training is not False and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)


        # convolve
        supports = list()
        for i in range(len(support_)):
            if not self.featureless: # if it has features x
                pre_sup = dot(x, self.weights_[i], sparse=self.is_sparse_inputs)
            else:
                pre_sup = self.weights_[i]

            support = dot(support_[i], pre_sup, sparse=True)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.bias

        return self.activation(output)
