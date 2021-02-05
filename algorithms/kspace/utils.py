import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops


def lr_scheduler(epoch, lr):
    """
    Use with LearningRateScheduler callback to dynamically adjust lr
    :param epoch:
    :param lr:
    :return:
    """
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def alpha_scheduler(epoch, alpha):
    """
    Use with AlphaRateScheduler callback to dynamically adjust alpha
    """
    return alpha * 0.1
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
        return np.array([np.exp((np.log(s_max + 1)/epochs)*i) for i in range(epochs)])
    else:
        return


class AlphaRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler.

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
        try:  # new API
            alpha = float(K.get_value(self.model.alpha))
            alpha = self.schedule(epoch, alpha)
        except TypeError:  # Support for old API for backward compatibility
            alpha = self.schedule(epoch)
        if not isinstance(alpha, (ops.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(alpha, ops.Tensor) and not alpha.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.alpha, K.get_value(alpha))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, alpha))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['alpha'] = K.get_value(self.model.alpha)