import tensorflow as tf
import numpy as np

loss_tracker = tf.keras.metrics.Mean(name="loss")
mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")


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
    def __init__(self, dims, output_dim, dropout):
        super(AE, self).__init__()

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

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def embed(self, x):
        return self.encoder(x, training=False)


class ClusterBooster(tf.keras.Model):
    """
    This module utilizes the cluster assignments to further optimize the model
    Specifically its loss function contains the minimization of KL-Divergence between a soft-clustering assignment
    distribution q and the target distribution p
    """
    def __init__(self, base_model, centers):
        super(ClusterBooster, self).__init__()

        self.pretrained = base_model
        self.centers = tf.Variable(initial_value=centers, trainable=True, dtype=tf.float32)


    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Compute the loss value. Note that the model is optimized according to the new loss only (KL-loss).
            # If a joint optimization of the kl-loss and the original mse loss is needed, it has to be implemented here
            # Also note that P and Q distributions are calculated via callback, on epoch begin.

            z = self.pretrained.encoder(data)
            z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]])  # reshape for broadcasting
            # UPDATE Q EVERY EPOCH
            partial = tf.math.pow(tf.norm(z - self.centers, axis=2, ord='euclidean'), 2)
            nominator = 1 / (1 + partial)
            denominator = tf.math.reduce_sum(1 / (1 + partial))
            self.Q = nominator / denominator

            partial = tf.math.pow(self.Q, 2) / tf.math.reduce_sum(self.Q, axis=1, keepdims=True)
            nominator = partial
            denominator = tf.math.reduce_sum(partial, axis=0)
            self.P = nominator / denominator

            loss = self.compiled_loss(self.P, self.Q)
        # Compute gradients
        gradients = tape.gradient(loss, self.pretrained.encoder.trainable_variables + [self.centers])
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.pretrained.encoder.trainable_variables + [self.centers]))

        loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}


    @property
    def metrics(self):

        return [loss_tracker]

    def embed(self, x):
        z = self.pretrained.embed(x)
        return z

    def call(self, x):
        return self.pretrained.call(x)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)
