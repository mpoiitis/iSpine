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


class DVAE(tf.keras.Model):
    """
    Denoising Autoencoder
    """
    def __init__(self, hidden1_dim, hidden2_dim, output_dim, dropout):
        super(DVAE, self).__init__()
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
            loss = self.compute_loss(x, y)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mse_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mse": mse_metric.result()}

    def compute_loss(self, x, y):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
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


class DAE(tf.keras.Model):
    def __init__(self, hidden1_dim, hidden2_dim, output_dim, dropout):
        super(DAE, self).__init__()
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

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = self.compute_loss(x, y)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mse_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mse": mse_metric.result()}

    def compute_loss(self, x, y):
        z = self.encoder(x)
        x_logit = self.decoder(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)

        return cross_ent

    @property
    def metrics(self):

        return [loss_tracker, mse_metric]

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def embed(self, x):
        return self.encoder(x, training=False)


class ClusterBooster(tf.keras.Model):
    """
    This module utilizes the pretrained neural network alongside the cluster assignments to further optimize the model
    Specifically its loss function contains a second part, apart from the pretrained model's loss, which is the minimization
    of KL-Divergence between a soft-clustering assignment distribution q and the target distribution p
    """
    def __init__(self, base_model, centers):
        super(ClusterBooster, self).__init__()

        self.pretrained = base_model
        self.centers = centers


    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.pretrained(x, training=True)  # Forward pass
            # Compute the original loss
            if self.pretrained.name == 'dae' or self.pretrained.name == 'dvae':
                original_loss = self.pretrained.compute_loss(x, y)
            elif self.pretrained.name == 'vae':
                original_loss = self.pretrained.compute_loss(x)
            else:  # ae
                mse = tf.keras.losses.MeanSquaredError()
                original_loss = mse(y, y_pred)

            # add the KL-divergence loss according to cluster assignments
            kl_loss = self.compute_loss(x)
            loss = original_loss + kl_loss

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mse_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mse": mse_metric.result()}

    def compute_loss(self, x):
        z = self.pretrained.embed(x)

        z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]]) # reshape for broadcasting

        partial = tf.norm(z - self.centers, axis=2, ord='euclidean')
        nominator = 1 / (1 + partial)
        denominator = tf.math.reduce_sum(1/ (1 + partial))
        self.Q = nominator / denominator


        partial =  tf.math.pow(self.Q, 2) / tf.math.reduce_sum(self.Q, axis=1, keepdims=True)
        nominator = partial
        denominator = tf.math.reduce_sum(partial, axis=0)
        P = nominator / denominator

        kl = tf.keras.losses.KLDivergence()

        return kl(P, self.Q)

    @property
    def metrics(self):

        return [loss_tracker, mse_metric]

    def embed(self, x):
        z = self.pretrained.embed(x)

        return z

    def call(self, x):
        return self.pretrained.call(x)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

