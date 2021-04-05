import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import os

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
    elif type == 'const':
        return [s_max] * epochs
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


def save_metrics(args, data, columns=['Acc', 'NMI', 'ARI', 'F1']):
    print(data, columns)
    args = vars(args)
    filename = ''
    for idx, (key, value) in enumerate(args.items()):
        if idx < len(args.keys()) - 1:
            if key == 'dims':
                filename += '{}_{}-'.format(key, '_'.join([str(v) for v in value]))
            else:
                filename += '{}_{}-'.format(key, value)
        else:
            filename += '{}_{}'.format(key, value)

    # write headers only once
    df = pd.DataFrame([data], columns=columns)
    if os.path.isfile('output/kSpaceGnn/{}.csv'.format(filename),):
        df.to_csv('output/kSpaceGnn/{}.csv'.format(filename), mode='a', index=None, header=False)
    else:
        df.to_csv('output/kSpaceGnn/{}.csv'.format(filename), mode='w', index=None, header=True)


def plot_metrics(filename='input_cora-method_kspace_gnn-repeats_10-dims_200_100-dropout_0.2-learning_rate_0.001-epochs_500-p_epochs_100-a_max_1.0-slack_400-alpha_const-save_True'):
    df = pd.read_csv('../../output/kSpaceGnn/{}.csv'.format(filename))

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
    plt.savefig('../../output/kSpaceGnn/figure_{}.png'.format(filename), format='png')
    plt.show()


def plot_metrics_per_layers():

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
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig('output/kSpaceGnn/figure_total_layers.png', format='png')
    plt.show()