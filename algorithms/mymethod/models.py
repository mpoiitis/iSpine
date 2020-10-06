import tensorflow as tf
import numpy as np

loss_tracker = tf.keras.metrics.Mean(name="loss")
mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

class VAE(tf.keras.Model):
    def __init__(self, hidden1_dim, hidden2_dim, output_dim, dropout):
        super(VAE, self).__init__()
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.hidden1_dim, activation=lrelu),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.hidden2_dim, activation=tf.nn.sigmoid),
            tf.keras.layers.Dropout(self.dropout)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden1_dim, activation=lrelu),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.output_dim, activation=tf.nn.sigmoid),
        ])


    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
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
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = self.compute_loss(x)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mse_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mse": mse_metric.result()}

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        kl_loss = -0.5 * tf.reduce_mean(logvar - tf.square(mean) - tf.exp(logvar) + 1)
        return cross_ent + kl_loss

    @property
    def metrics(self):
        """ We list our `Metric` objects here so that `reset_states()` can be
            called automatically at the start of each epoch
            or at the start of `evaluate()`.
            If you don't implement this property, you have to call
            `reset_states()` yourself at the time of your choosing.
        """
        return [loss_tracker, mse_metric]

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        decoded = self.decode(z)
        return decoded

    def embed(self, x):
        return self.encoder(x, training=False)


class AE(tf.keras.Model):
    def __init__(self, hidden1_dim, hidden2_dim, output_dim, dropout):
        super(AE, self).__init__()
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.hidden1_dim, activation=lrelu),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.hidden2_dim, activation=tf.nn.sigmoid),
            tf.keras.layers.Dropout(self.dropout)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden1_dim, activation=lrelu),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.output_dim, activation=tf.nn.sigmoid),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def embed(self, x):
        return self.encoder(x, training=False)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)
