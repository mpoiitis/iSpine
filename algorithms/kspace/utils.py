import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from utils.metrics import clustering_metrics


def lrelu(x, leak=0.2, name="lrelu"):
    """
    Leaky ReLU function
    """
    return tf.math.maximum(x, leak * x)


def get_alpha(s_max, epochs, type='linear'):
    """
    Calculate evenly spaced alphas for each epoch based on function type
    :param s_max: the maximum value of a. Last epoch will have this value
    :param epochs: number of epochs
    :param type: function type
    :return: a numpy array of shape (epochs, 1)
    """
    if type == 'linear':
        return np.linspace(0, s_max, epochs)
    elif type == 'exp':
        zeros = [0] * 50
        return np.array(zeros + [np.exp((np.log(s_max + 1) / epochs) * i) for i in range(1, epochs-49)])
    elif type == 'zeros':
        return [0] * epochs
    else:
        return


def cluster_loss(z, centers):
    z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]])

    centers = tf.reshape(centers, [1, tf.shape(centers)[0], tf.shape(centers)[1]])
    partial = tf.math.pow(tf.squeeze(tf.norm(z - centers, ord='euclidean', axis=2)), 2)
    nominator = 1 / (1 + partial)
    denominator = tf.math.reduce_sum(1 / (1 + partial), axis=1)
    denominator = tf.reshape(denominator, [tf.shape(denominator)[0], 1])
    q = nominator / denominator
    q_norm = 1 - q
    q_norm = tf.math.log(q_norm + 0.00001)  # e for 0 logs
    return tf.reduce_sum(q_norm, axis=1)


class ClusterLoss(LossFunctionWrapper):

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='cluster_loss'):
        super(ClusterLoss, self).__init__(cluster_loss, name=name, reduction=reduction)


def assign_clusters(z, c, gnd):
    z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]])  # reshape for broadcasting
    c = tf.reshape(c, [1, tf.shape(c)[0], tf.shape(c)[1]])  # from (batch_size, num_centers*emb_dim) to (batch_size, num_centers, emb_dim)

    diffs = tf.squeeze(tf.norm(z - c, ord='euclidean', axis=2))
    predicted = tf.argmin(diffs, axis=1)  # for every instance find the cluster with the smallest difference

    predicted = predicted.numpy()

    cm = clustering_metrics(gnd, predicted)
    acc, nmi, f1_macro, ari = cm.evaluationClusterModelFromLabel()

    return acc, nmi, f1_macro, ari
