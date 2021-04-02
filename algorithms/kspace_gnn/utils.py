import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

EPS = 1e-15

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
        return np.array(zeros + [s_max * (np.exp(0.025*x) - 1) for x in range(0, epochs - slack + 1)])
    elif type == 'zeros':
        return [0] * epochs
    elif type == 'ones':
        return [1] * epochs
    else:
        return


def cluster_q_loss(z, centers):
    z = z.view(z.size()[0], 1, z.size()[1])
    centers_r = centers.view(1, centers.size()[0], centers.size()[1])
    partial = torch.squeeze(torch.norm(z - centers_r, p=2, dim=2)) ** 2
    nominator = 1 - (1 / (1 + partial))
    denominator = torch.sum(nominator, dim=1)
    denominator = denominator.view(denominator.size()[0], 1)
    q = nominator / denominator
    q_log = torch.log2(q + 0.00001)  # e for 0 logs
    return torch.sum(q_log, dim=1) # + tf.reduce_sum(tf.norm(centers, ord='euclidean', axis=1))


def cluster_kl_loss(q):
    # p_nom = (q ** 2) / torch.sum(q ** 2, dim=0)
    # p_denom = torch.sum(p_nom, dim=1)
    # p_denom = p_denom.view(p_denom.size()[0], 1)
    # p = p_nom / p_denom
    p_nom = q / torch.sqrt(torch.sum(q ** 2, dim=0))
    p_denom = torch.sum(p_nom, dim=1)
    p_denom = p_denom.view(p_denom.size()[0], 1)
    p = p_nom / p_denom

    # return torch.mean(torch.sum((-q.log() * p), dim=1))
    return F.kl_div(torch.log2(p + EPS), q)


def calc_metrics(y_pred, y_true):
    # assert y_pred.size == y_true.size
    print('Predicted: {}, Actual: {}'.format(len(np.unique(y_pred)), len(np.unique(y_true))))
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # map predicted values to the correct ones
    y_pred = [col_ind[row_ind.tolist().index(l)] for l in y_pred]

    # calculate metrics
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')

    return acc, nmi, ari, f1_macro