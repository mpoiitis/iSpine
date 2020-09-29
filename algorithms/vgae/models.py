import tensorflow as tf
from .layers import VariationalEncoder, Encoder, Decoder
from utils.metrics import masked_sigmoid_cross_entropy


class VGAE(tf.keras.Model):
    """variational graph autoencoder."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_features_nonzero, dropout, is_sparse_inputs, **kwargs):
        super(VGAE, self).__init__(**kwargs)

        self.input_dim = input_dim  # 1433
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        self.encoder = VariationalEncoder(input_dim=self.input_dim, output_dim=self.output_dim, hidden_dim=self.hidden_dim,
                                num_features_nonzero=num_features_nonzero, activation=tf.nn.relu, dropout=self.dropout, is_sparse_inputs=is_sparse_inputs)
        self.decoder = Decoder(input_dim=self.output_dim, dropout=self.dropout)

    def call(self, inputs, training=None):
        x, mask, adj = inputs

        z_mean, z_log_var, z = self.encoder((x, adj), training)
        reconstructed = self.decoder(z)

        # convert adjacency to dense and flatten to feed it in the loss calculation function
        adj_dense = tf.sparse.to_dense(tf.sparse.reorder(adj[0]))
        adj_dense = tf.reshape(adj_dense, [-1])

        loss = tf.zeros([])
        for var in self.trainable_variables:
            loss += tf.nn.l2_loss(var)

        # Cross entropy error
        loss += masked_sigmoid_cross_entropy(reconstructed, adj_dense, mask)

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        loss += kl_loss
        return loss

    def embed(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs, False)
        return z


class GAE(tf.keras.Model):
    """ Graph autoencoder"""

    def __init__(self, input_dim, output_dim, hidden_dim, num_features_nonzero, dropout, is_sparse_inputs, **kwargs):
        super(GAE, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.encoder = Encoder(input_dim=self.input_dim, output_dim=self.output_dim, num_features_nonzero=num_features_nonzero,
                               activation=tf.nn.relu, dropout=self.dropout, is_sparse_inputs=is_sparse_inputs)
        self.decoder = Decoder(input_dim=self.output_dim, dropout=self.dropout)


    def call(self, inputs, training=None):
        x, mask, adj = inputs
        z = self.encoder((x, adj), training)
        reconstructed = self.decoder(z)

        # convert adjacency to dense and flatten to feed it in the loss calculation function
        adj_dense = tf.sparse.to_dense(tf.sparse.reorder(adj[0]))
        adj_dense = tf.reshape(adj_dense, [-1])

        loss = tf.zeros([])
        for var in self.trainable_variables:
            loss += tf.nn.l2_loss(var)

        # Cross entropy error
        loss += masked_sigmoid_cross_entropy(reconstructed, adj_dense, mask)

        return loss

    def embed(self, inputs):
        z = self.encoder(inputs, False)
        return z