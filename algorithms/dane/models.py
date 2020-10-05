import tensorflow as tf
from algorithms.dane.utils.utils import lrelu


class AutoEncoder(tf.keras.Model):
    def __init__(self, hidden1_dim, hidden2_dim, output_dim, dropout):
        super(AutoEncoder, self).__init__()
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


class FullModel(tf.keras.Model):
    def __init__(self, net_model, att_model, config):
        super(FullModel, self).__init__()

        self.config = config
        self.net_model = net_model
        self.att_model = att_model
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.m_autoencoder = self.net_model
        self.z_autoencoder = self.att_model

    def call(self, inputs):
        m, neg_m, z, neg_z, w, neg_w = inputs

        # for reconstruction loss
        m_pred = self.m_autoencoder(m)
        z_pred = self.z_autoencoder(z)
        neg_m_pred = self.m_autoencoder(neg_m)
        neg_z_pred = self.z_autoencoder(neg_z)

        # for cross-modality and first-order proximity
        h_m = self.m_autoencoder.embed(m)
        h_z = self.z_autoencoder.embed(z)
        h_neg_m = self.m_autoencoder.embed(neg_m)
        h_neg_z = self.z_autoencoder.embed(neg_z)

        # ===============reconstruction==================
        mse = tf.keras.losses.MeanSquaredError()
        rec_loss_m = mse(m, m_pred)
        rec_loss_neg_m = mse(neg_m, neg_m_pred)
        rec_loss_z = mse(z, z_pred)
        rec_loss_neg_z = mse(neg_z, neg_z_pred)
        recon_loss = rec_loss_m + rec_loss_neg_m + rec_loss_z + rec_loss_neg_z

        # ===============cross modality proximity==================
        pre_logit_pos = tf.reduce_sum(tf.multiply(h_m, h_z), 1)
        pre_logit_neg_1 = tf.reduce_sum(tf.multiply(h_neg_m, h_z), 1)
        pre_logit_neg_2 = tf.reduce_sum(tf.multiply(h_m, h_neg_z), 1)

        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pre_logit_pos), logits=pre_logit_pos)
        neg_loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_1),
                                                             logits=pre_logit_neg_1)
        neg_loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_2),
                                                             logits=pre_logit_neg_2)
        cross_modal_loss = tf.reduce_mean(pos_loss + neg_loss_1 + neg_loss_2)

        # =============== first-order proximity================
        pre_logit_pp_x = tf.matmul(h_m, h_m, transpose_b=True)
        pre_logit_pp_z = tf.matmul(h_z, h_z, transpose_b=True)
        pre_logit_nn_x = tf.matmul(h_neg_m, h_neg_m, transpose_b=True)
        pre_logit_nn_z = tf.matmul(h_neg_z, h_neg_z, transpose_b=True)

        pp_x_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=w + tf.eye(tf.shape(w)[0]), logits=pre_logit_pp_x) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.linalg.diag_part(pre_logit_pp_x)),
                                                              logits=tf.linalg.diag_part(pre_logit_pp_x))
        pp_z_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=w + tf.eye(tf.shape(w)[0]), logits=pre_logit_pp_z) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.linalg.diag_part(pre_logit_pp_z)),
                                                              logits=tf.linalg.diag_part(pre_logit_pp_z))

        nn_x_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=w + tf.eye(tf.shape(neg_w)[0]),
                                                            logits=pre_logit_nn_x) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.linalg.diag_part(pre_logit_nn_x)),
                                                              logits=tf.linalg.diag_part(pre_logit_nn_x))
        nn_z_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=w + tf.eye(tf.shape(neg_w)[0]),
                                                            logits=pre_logit_nn_z) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.linalg.diag_part(pre_logit_nn_z)),
                                                              logits=tf.linalg.diag_part(pre_logit_nn_z))
        first_order_loss = tf.reduce_mean(pp_x_loss + pp_z_loss + nn_x_loss + nn_z_loss)

        loss = recon_loss * self.beta + first_order_loss * self.gamma + cross_modal_loss * self.alpha

        return loss

    def embed(self, inputs):
        m, z = inputs
        h_m = self.net_model.embed(m)
        h_z = self.att_model.embed(z)
        H = tf.concat([tf.compat.v1.nn.l2_normalize(h_m, dim=1), tf.compat.v1.nn.l2_normalize(h_z, dim=1)], axis=1)

        return h_m, h_z, H