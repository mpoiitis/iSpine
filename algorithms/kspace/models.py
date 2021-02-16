import tensorflow as tf
import numpy as np
from .utils import lrelu, ClusterLoss

loss_tracker = tf.keras.metrics.Mean(name="loss")
mse_loss = tf.keras.losses.MeanSquaredError()
cluster_loss = ClusterLoss()
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


class AE(tf.keras.Model):
    def __init__(self, dims, output_dim, dropout, num_centers, alphas):
        """
        :param dims: list of integers corresponding to each layer's dimensions
        :param output_dim: int. the dimension of the reconstructed output
        :param dropout: float. dropout rate
        :param num_centers: int, the number of centers to calculate
        :param alphas: np.array with the alpha rate for every epoch. Used in callback to update alpha
        """
        super(AE, self).__init__()

        self.dims = dims[:] # the slice operator means that this is a shallow copy. Note the dims.reverse() below. self.dims needs the original list
        self.num_centers = num_centers
        latent_dim = self.dims[-1]

        layers = len(dims)

        encoder_layers = list()
        for i in range(layers):
            encoder_layers.append(tf.keras.layers.Dropout(dropout))
            if i != layers - 1:
                activation = lrelu
                encoder_layers.append(tf.keras.layers.Dense(dims[i], activation=activation, kernel_initializer=initializer))
            else:
                activation = None
                encoder_layers.append(tf.keras.layers.Dense(latent_dim, activation=activation, kernel_initializer=initializer))

        cluster_layers = list()
        cluster_layers.append(tf.keras.layers.Dropout(dropout))
        cluster_layers.append(tf.keras.layers.Dense(self.num_centers*latent_dim, activation=lrelu, kernel_initializer=initializer))

        dims.reverse()
        decoder_layers = list()
        for i in range(1, layers + 1):
            decoder_layers.append(tf.keras.layers.Dropout(dropout))
            if i != layers:
                activation = lrelu
                decoder_layers.append(tf.keras.layers.Dense(dims[i], activation=activation, kernel_initializer=initializer))
            else:
                activation = None  # linear last layer
                decoder_layers.append(tf.keras.layers.Dense(output_dim, activation=activation, kernel_initializer=initializer))

        self.encoder = tf.keras.Sequential(encoder_layers)
        self.cluster_generator = tf.keras.Sequential(cluster_layers)
        self.decoder = tf.keras.Sequential(decoder_layers)

        self.alphas = tf.convert_to_tensor(alphas, dtype=tf.float32, name='alphas')
        self.alpha = tf.Variable(0, trainable=False, dtype=tf.float32)

        # self.centers = tf.random.normal([num_centers, latent_dim], mean=0.0, stddev=1.0, dtype=tf.float32, seed=123)
        
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z = self.encoder(x)
            y_pred = self.decoder(z)

            centers = self.cluster_generator(z)
            centers = tf.reshape(centers, [tf.shape(centers)[0], self.num_centers, -1]) # batch size x (centers*dim) -> batch size x centers x dim
            self.centers = tf.reduce_mean(centers, axis=0)  # mean across batch dim. batch size x centers x dim -> 1 x centers x dim
            # MSE + the Q optimization loss with alpha regularization factors
            rec_loss = mse_loss(y, y_pred)
            c_loss = cluster_loss(z, self.centers)
            loss = rec_loss + self.alpha * c_loss

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        loss_tracker.update_state(loss)
        return {'loss': loss_tracker.result(), 'Reconstruction': rec_loss, 'Clustering': c_loss}

    def call(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded

    def embed(self, x):
        return self.encoder(x, training=False)



