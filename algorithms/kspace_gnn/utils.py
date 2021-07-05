import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import os

EPS = 1e-15


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


def plot_metrics(filename='input_cora-method_kspace_gnn-repeats_10-dims_200_100-dropout_0.2-learning_rate_0.001-epochs_500-p_epochs_100-a_max_1.0-slack_400-alpha_const-save_True'):
    df = pd.read_csv('output/kSpaceGnn/{}.csv'.format(filename))

    acc_std = df['Acc'].std()
    nmi_std = df['NMI'].std()
    ari_std = df['ARI'].std()
    f1_std = df['F1'].std()
    plt.plot(df.index, df['Acc'], c='red', label='Acc', linestyle='-')
    plt.fill_between(df.index, df['Acc'] - acc_std, df['Acc'] + acc_std, alpha=.2, color='red')
    plt.plot(df.index, df['NMI'], c='blue', label='NMI', linestyle='--')
    plt.fill_between(df.index, df['NMI'] - nmi_std, df['NMI'] + nmi_std, alpha=.2, color='blue')
    plt.plot(df.index, df['ARI'], c='green', label='ARI', linestyle='-.')
    plt.fill_between(df.index, df['ARI'] - ari_std, df['ARI'] + ari_std, alpha=.2, color='green')
    plt.plot(df.index, df['F1'], c='purple', label='F1')
    plt.fill_between(df.index, df['F1'] - f1_std, df['F1'] + f1_std, alpha=.2, color='purple')
    plt.plot(df.index, [0.768] * len(df.index), color='black', label='base_Acc', linestyle='-')
    plt.plot(df.index, [0.607] * len(df.index), color='black', label='base_NMI', linestyle='--')
    plt.plot(df.index, [0.565] * len(df.index), color='black', label='base_ARI', linestyle='-.')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
    plt.xlabel('Iteration')
    plt.ylabel('Value')

    plt.gca().yaxis.set_ticks(np.arange(0, 1, 0.1))
    plt.savefig('output/kSpaceGnn/figure_{}.png'.format(filename), format='png')
    plt.show()


def plot_metrics_for_layers():

    directory = 'output/kSpaceGnn/'
    acc_stds = list()
    acc_means = list()
    nmi_stds = list()
    nmi_means = list()
    ari_stds = list()
    ari_means = list()
    f1_stds = list()
    f1_means = list()
    layers = list()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            dims = filename.split('dims_')[1].split('-dropout')[0]
            num_layers = len(dims.split('_'))
            layers.append(num_layers)

            df = pd.read_csv(directory + filename)
            acc_stds.append(df['Acc'].std())
            acc_means.append(df['Acc'].mean())
            nmi_stds.append(df['NMI'].std())
            nmi_means.append(df['NMI'].mean())
            ari_stds.append(df['ARI'].std())
            ari_means.append(df['ARI'].mean())
            f1_stds.append(df['F1'].std())
            f1_means.append(df['F1'].mean())

    acc_stds = np.array(acc_stds)
    acc_means = np.array(acc_means)
    nmi_stds = np.array(nmi_stds)
    nmi_means = np.array(nmi_means)
    ari_stds = np.array(ari_stds)
    ari_means = np.array(ari_means)
    f1_stds = np.array(f1_stds)
    f1_means = np.array(f1_means)

    plt.plot(layers, acc_means, c='red', label='Acc', linestyle='-')
    plt.fill_between(layers, acc_means - acc_stds, acc_means + acc_stds, alpha=.2, color='red')
    plt.plot(layers, nmi_means, c='blue', label='NMI', linestyle='--')
    plt.fill_between(layers, nmi_means - nmi_stds, nmi_means + nmi_stds, alpha=.2, color='blue')
    plt.plot(layers, ari_means, c='green', label='ARI', linestyle='-.')
    plt.fill_between(layers, ari_means - ari_stds, ari_means + ari_stds, alpha=.2, color='green')
    plt.plot(layers, f1_means, c='purple', label='F1')
    plt.fill_between(layers, f1_means - f1_stds, f1_means + f1_stds, alpha=.2, color='purple')
    plt.plot(layers, [0.768] * len(layers), color='black', label='base_Acc', linestyle='-')
    plt.plot(layers, [0.607] * len(layers), color='black', label='base_NMI', linestyle='--')
    plt.plot(layers, [0.565] * len(layers), color='black', label='base_ARI', linestyle='-.')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
    plt.xlabel('Layers')
    plt.ylabel('Value')
    plt.gca().yaxis.set_ticks(np.arange(0, 1, 0.1))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig('output/kSpaceGnn/figure_total_layers.png', format='png')
    plt.show()