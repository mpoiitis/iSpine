import tensorflow as tf
import numpy as np
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def lrelu(x, leak=0.2, name="lrelu"):
    """
    Leaky ReLU function
    """
    return tf.math.maximum(x, leak * x)


def get_alpha(s_max, epochs, slack, type='linear'):
    """
    Calculate evenly spaced alphas for each epoch based on function type
    :param s_max: the maximum value of a. Last epoch will have this value
    :param epochs: number of epochs
    :param slack: number of zeros for the first epochs
    :param type: function type
    :return: a numpy array of shape (epochs, 1)
    """
    if type == 'linear':
        zeros = [0] * slack
        return np.array(zeros + np.linspace(0, s_max, epochs - slack).tolist())
    elif type == 'exp':
        zeros = [0] * slack
        return np.array(zeros + [np.exp((np.log(s_max + 1) / epochs) * i) for i in range(1, epochs - slack + 1)])
    elif type == 'zeros':
        return [0] * epochs
    else:
        return


def cluster_loss(z, centers):
    z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]])

    centers_r = tf.reshape(centers, [1, tf.shape(centers)[0], tf.shape(centers)[1]])
    partial = tf.math.pow(tf.squeeze(tf.norm(z - centers_r, ord='euclidean', axis=2)), 2)
    nominator = 1 / (1 + partial)
    denominator = tf.math.reduce_sum(1 / (1 + partial), axis=1)
    denominator = tf.reshape(denominator, [tf.shape(denominator)[0], 1])
    q = nominator / denominator
    q_norm = 1 - q
    q_norm = tf.math.log(q_norm + 0.00001)  # e for 0 logs
    return tf.reduce_sum(q_norm, axis=1) + tf.reduce_sum(tf.norm(centers, ord='euclidean', axis=1))


class ClusterLoss(LossFunctionWrapper):

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='cluster_loss'):
        super(ClusterLoss, self).__init__(cluster_loss, name=name, reduction=reduction)


def calc_metrics(Y_pred, Y):
    assert Y_pred.size == Y.size
    # assert len(np.unique(Y_pred)) == len(np.unique(Y))
    print('Predicted: {}, Actual: {}'.format(len(np.unique(Y_pred)), len(np.unique(Y))))
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # map predicted values to the correct ones
    Y_pred = [col_ind[row_ind.tolist().index(l)] for l in Y_pred]

    # calculate metrics
    nmi = metrics.normalized_mutual_info_score(Y, Y_pred)
    ari = metrics.adjusted_rand_score(Y, Y_pred)
    acc = metrics.accuracy_score(Y, Y_pred)
    f1_macro = metrics.f1_score(Y, Y_pred, average='macro')

    return acc, nmi, ari, f1_macro