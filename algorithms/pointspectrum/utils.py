import csv
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

EPS = 1e-15


def get_hyperparams(s_max, epochs, type='linear', increasing=True):
    """
    Calculate evenly spaced hyperparam values for each epoch based on function type
    :param s_max: the maximum value.
    :param epochs: number of epochs
    :param type: function type
    :param increasing: if true, the values will be in increasing order, else decreasing
    """
    if type == 'linear':
        values = np.linspace(0, s_max, epochs).tolist()
    elif type == 'exp':
        # return [s_max * (np.exp(0.025*x) - 1) for x in range(epochs)]
        values = [(np.exp((np.log(1+s_max)/epochs) * x) - 1) for x in range(epochs)]
    elif type == 'const':
        values = [s_max] * epochs
    else:
        return

    if increasing:
        return values
    else:
        return values[::-1]


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
    # print('Predicted: {}, Actual: {}'.format(len(np.unique(y_pred)), len(np.unique(y_true))))
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


def write_to_csv(args, best_epoch, best_acc, best_nmi, best_ari, best_f1, total_time):
    directory = 'output/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = '{}/results.csv'.format(directory)
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        headers = ['Dataset', 'Dims', 'Dropout', 'Epochs', 'lr', 'Power', 'Temperature', 'Alpha', 'Alpha fun', 'Beta', 'Beta fun', 'Acc', 'Nmi', 'Ari', 'F1', 'Best Epoch', 'Total Time']
        writer = csv.DictWriter(file, delimiter=',', fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        dims = '_'.join([str(v) for v in args.dims])
        writer.writerow({'Dataset': args.input, 'Dims': dims, 'Dropout': args.dropout, 'Epochs': args.epochs, 'lr': args.learning_rate,
                         'Power': args.power, 'Temperature': args.temperature, 'Alpha': args.alpha, 'Alpha fun': args.a_prog, 'Beta': args.beta,
                         'Beta fun': args.b_prog, 'Acc': best_acc, 'Nmi': best_nmi, 'Ari': best_ari, 'F1': best_f1, 'Best Epoch': best_epoch, 'Total Time': total_time})

    return


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data = self.x[idx, :]
        target = self.y[idx, :]
        return data, target
