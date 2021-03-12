import tensorflow as tf
import numpy as np
from .utils import lrelu

loss_tracker = tf.keras.metrics.Mean(name="loss")
mse_loss = tf.keras.losses.MeanSquaredError()
initializer = tf.keras.initializers.GlorotNormal

class VAE(tf.keras.Model):
    def __init__(self, dims, output_dim, dropout):
        super(VAE, self).__init__()

        layers = len(dims)

        encoder_layers = list()
        for i in range(layers):
            encoder_layers.append(tf.keras.layers.Dropout(dropout))
            if i != layers - 1:
                activation = lrelu
            else:
                activation = tf.nn.sigmoid
            encoder_layers.append(tf.keras.layers.Dense(dims[i], activation=activation))

        dims.reverse()
        decoder_layers = list()
        for i in range(1, layers + 1):
            decoder_layers.append(tf.keras.layers.Dropout(dropout))
            if i != layers:
                activation = lrelu
                decoder_layers.append(tf.keras.layers.Dense(dims[i], activation=activation))
            else:
                activation = tf.nn.sigmoid
                decoder_layers.append(tf.keras.layers.Dense(output_dim, activation=activation))

        self.encoder = tf.keras.Sequential(encoder_layers)
        self.decoder = tf.keras.Sequential(decoder_layers)

    def encode(self, x, training=True):
        mean, logvar = tf.split(self.encoder(x, training=training), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.math.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.math.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x):
        """
        Minimize ELBO on the marginal log-likelihood
        :param x:
        :return:
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

        logpx_z = -tf.math.reduce_sum(cross_ent)
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def embed(self, x):
        mean, logvar = self.encode(x, training=False)
        z = self.reparameterize(mean, logvar)
        return z


class Encoder(tf.keras.Model):
    def __init__(self, dims, dropout):
        super(Encoder, self).__init__()

        encoder_layers = list()
        for i in range(len(dims)):
            encoder_layers.append(tf.keras.layers.Dropout(dropout))
            encoder_layers.append(tf.keras.layers.Dense(dims[i], activation=tf.nn.relu))

        self.encoder = tf.keras.Sequential(encoder_layers)

    def call(self, x):
        e = self.encoder(x)
        return e


class Decoder(tf.keras.Model):
    def __init__(self, dims, output_dim, dropout):
        super(Decoder, self).__init__()

        layers = len(dims)

        dims = dims[::-1]  # reverse dimensions for the decoder
        decoder_layers = list()
        for i in range(1, layers + 1):
            decoder_layers.append(tf.keras.layers.Dropout(dropout))
            if i != layers:
                decoder_layers.append(tf.keras.layers.Dense(dims[i], activation=tf.nn.relu))
            else:  # linear last layer
                decoder_layers.append(tf.keras.layers.Dense(output_dim, activation=None))

        self.decoder = tf.keras.Sequential(decoder_layers)

    def call(self, x):
        y = self.decoder(x)
        # return tf.keras.activations.sigmoid(y)
        return y


class AE(tf.keras.Model):
    def __init__(self, dims, output_dim, dropout, num_centers):
        """
        :param dims: list of integers corresponding to each layer's dimensions
        :param output_dim: int. the dimension of the reconstructed output
        :param dropout: float. dropout rate
        :param num_centers: int, the number of centers to calculate
        """
        super(AE, self).__init__()

        self.num_centers = num_centers
        self.encoder = Encoder(dims, dropout)
        self.decoder = Decoder(dims, output_dim, dropout)

        embedding_dim = dims[-1]
        centers = tf.random.normal((7, embedding_dim), mean=0.0, stddev=1.0)
        self.centers = tf.Variable(centers, trainable=True, dtype=tf.float32, name='centers')

    def call(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded

    def predict(self, x):
        z = self.encoder(x, training=False)
        z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]])
        centers = tf.reshape(self.centers, [1, tf.shape(self.centers)[0], tf.shape(self.centers)[1]])

        partial = tf.math.pow(tf.squeeze(tf.norm(z - centers, ord='euclidean', axis=2)), 2)
        nominator = (1 / (1 + partial))
        denominator = tf.math.reduce_sum(nominator, axis=1)
        denominator = tf.reshape(denominator, [tf.shape(denominator)[0], 1])
        q = nominator / denominator
        return tf.argmax(q, axis=1)

    def embed(self, x):
        return self.encoder(x, training=False)



