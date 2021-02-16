import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from utils.metrics import clustering_metrics
from utils.plots import plot_centers
import datetime


def lrelu(x, leak=0.2, name="lrelu"):
    """
    Leaky ReLU function
    """
    return tf.math.maximum(x, leak * x)


def lr_scheduler(epoch, lr):
    """
    Use with LearningRateScheduler callback to dynamically adjust lr
    """
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def alpha_scheduler(epoch, alphas):
    """
    Use with AlphaRateScheduler callback to dynamically adjust alpha
    """
    return alphas[epoch]
    # return alpha * tf.math.exp(0.1)


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
        return np.array([np.exp((np.log(s_max + 1) / epochs) * i) for i in range(epochs)])
    else:
        return


class AlphaRateScheduler(tf.keras.callbacks.Callback):
    """Alpha rate scheduler.

      At the beginning of every epoch, this callback gets the updated alpha rate
      value from `schedule` function provided at `__init__`, with the current epoch
      and current alpha rate, and applies the updated alpha rate on the model.

      Arguments:
        schedule: a function that takes an epoch index (integer, indexed from 0)
            and current alpha rate (float) as inputs and returns a new alpha rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(AlphaRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, 'alpha'):
            raise ValueError('Model must have a "alpha" attribute.')

        alphas = np.array(K.get_value(self.model.alphas))
        alpha = self.schedule(epoch, alphas)

        if not isinstance(alpha, (ops.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(alpha, ops.Tensor) and not alpha.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.alpha, K.get_value(alpha))
        if self.verbose > 0:
            print('\nEpoch %d: AlphaRateScheduler changing alpha to %s ' % (epoch + 1, alpha))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['alpha'] = K.get_value(self.model.alpha)


class PlotCallback(tf.keras.callbacks.Callback):

    def __init__(self, gnd, data, verbose=0):
        super(PlotCallback, self).__init__()
        self.gnd = gnd
        self.data = data
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):

        if epoch == 0 or (epoch + 1) % 50 == 0:
            z = self.model.embed(self.data)
            centers = self.model.centers
            plot_centers(z, centers, self.gnd, epoch)


def cluster_loss(z, centers):
    z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]])

    centers = tf.reshape(centers, [1, tf.shape(centers)[0], tf.shape(centers)[1]])
    # partial = tf.math.pow(tf.squeeze(tf.norm(z - centers, ord='euclidean', axis=2)), 2)
    # nominator = tf.math.reciprocal(1 + partial)
    # denominator = tf.math.reduce_sum(tf.math.reciprocal(1 + partial), axis=1)
    # denominator = tf.reshape(denominator, [tf.shape(denominator)[0], 1])
    # # q_norm = K.l2_normalize(q, axis=1)
    # q = nominator / denominator
    # q_norm = 1 - q
    # q_norm = tf.math.log(q_norm + 0.00001)  # e for 0 logs
    # return tf.reduce_sum(q_norm, axis=1)

    q = tf.math.pow(tf.squeeze(tf.norm(z - centers, ord='euclidean', axis=2)), 2)
    q = tf.math.log(q + 0.00001)

    return tf.reduce_sum(q, axis=1)


class ClusterLoss(LossFunctionWrapper):

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='cluster_loss'):
        super(ClusterLoss, self).__init__(cluster_loss, name=name, reduction=reduction)


def l1_normalize(x, axis=None, epsilon=1e-12, name=None):
    with ops.name_scope(name, "l1_normalize", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        square_sum = math_ops.reduce_sum(math_ops.abs(x), axis, keepdims=True)
        x_inv_norm = math_ops.reciprocal(math_ops.maximum(square_sum, epsilon))
        return math_ops.multiply(x, x_inv_norm, name=name)


def assign_clusters(z, c, gnd):
    m = len(np.unique(gnd))
    tf.print(tf.shape(z))
    tf.print(tf.shape(c))
    z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]])  # reshape for broadcasting
    c = tf.reshape(c, [tf.shape(c)[0], m, -1])  # from (batch_size, num_centers*emb_dim) to (batch_size, num_centers, emb_dim)

    diffs = tf.squeeze(tf.norm(z - c, ord='euclidean', axis=2))
    predicted = tf.argmin(diffs, axis=1)  # for every instance find the cluster with the smallest difference

    predicted = predicted.numpy()

    from collections import Counter
    c = Counter(list(predicted))
    print("Predicted: ", c)
    c = Counter(list(gnd))
    print("Ground Truth: ",  c)

    cm = clustering_metrics(gnd, predicted)
    acc, nmi, f1_macro, ari = cm.evaluationClusterModelFromLabel()

    return acc, nmi, f1_macro, ari