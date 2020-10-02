import tensorflow as tf


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)
