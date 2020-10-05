import tensorflow as tf
import numpy as np
from sklearn import preprocessing

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def generate_samples(walks, features, adj, model, batch_size, num_nodes):
    X = []
    Z = []

    order = np.arange(num_nodes)
    np.random.shuffle(order)

    index = 0
    while True:
        if index > num_nodes:
            break
        if index + batch_size < num_nodes:
            mini_batch = sample_by_idx(order[index:index + batch_size], walks, features, adj)
        else:
            mini_batch = sample_by_idx(order[index:], walks, features, adj)
        index += batch_size

        net_H, att_H, _ = model.embed((mini_batch.X, mini_batch.Z))
        X.extend(net_H.numpy())
        Z.extend(att_H.numpy())

    X = np.array(X)
    Z = np.array(Z)

    X = preprocessing.normalize(X, norm='l2')
    Z = preprocessing.normalize(Z, norm='l2')

    sim = np.dot(X, Z.T)
    neg_idx = np.argmin(sim, axis=1)

    return order, neg_idx

def sample_by_idx(idx, walks, features, adj):
    mini_batch = Dotdict()
    mini_batch.X = walks[idx]
    mini_batch.Z = features[idx]
    mini_batch.W = adj[idx][:, idx]

    return mini_batch