import scipy.io as sio
import time
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from utils.metrics import clustering_metrics, square_dist


def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


def run_agc(args):

    dataset = args.input
    data = sio.loadmat('data/agc_data/{}.mat'.format(dataset))
    feature = data['fea']
    if sp.issparse(feature):
        feature = feature.todense()

    adj = data['W']
    gnd = data['gnd']
    gnd = gnd.T
    gnd = gnd - 1
    gnd = gnd[0, :]
    k = len(np.unique(gnd))
    adj = sp.coo_matrix(adj)
    intra_list = []
    intra_list.append(10000)


    acc_list = []
    nmi_list = []
    f1_list = []
    stdacc_list = []
    stdnmi_list = []
    stdf1_list = []
    max_iter = args.max_iter
    rep = 10
    t = time.time()
    adj_normalized = preprocess_adj(adj)
    adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2


    tt = 0
    while 1:
        tt = tt + 1
        power = tt

        intraD = np.zeros(rep)
        ac = np.zeros(rep)
        nm = np.zeros(rep)
        f1 = np.zeros(rep)

        feature = adj_normalized.dot(feature)

        u, s, v = sp.linalg.svds(feature, k=k, which='LM')  # matrix u of SVD is equal to calculating the kernel X*X_T

        for i in range(rep):
            kmeans = KMeans(n_clusters=k).fit(u)
            predict_labels = kmeans.predict(u)
            intraD[i] = square_dist(predict_labels, feature)
            cm = clustering_metrics(gnd, predict_labels)
            ac[i], nm[i], f1[i] = cm.evaluationClusterModelFromLabel()

        intramean = np.mean(intraD)
        acc_means = np.mean(ac)
        acc_stds = np.std(ac)
        nmi_means = np.mean(nm)
        nmi_stds = np.std(nm)
        f1_means = np.mean(f1)
        f1_stds = np.std(f1)

        intra_list.append(intramean)
        acc_list.append(acc_means)
        stdacc_list.append(acc_stds)
        nmi_list.append(nmi_means)
        stdnmi_list.append(nmi_stds)
        f1_list.append(f1_means)
        stdf1_list.append(f1_stds)
        print('power: {}'.format(power),
              'intra_dist: {}'.format(intramean),
              'acc_mean: {}'.format(acc_means),
              'acc_std: {}'.format(acc_stds),
              'nmi_mean: {}'.format(nmi_means),
              'nmi_std: {}'.format(nmi_stds),
              'f1_mean: {}'.format(f1_means),
              'f1_std: {}'.format(f1_stds))

        if intra_list[tt] > intra_list[tt - 1] or tt > max_iter:
            print('bestpower: {}'.format(tt - 1))
            t = time.time() - t
            print('time:', t)
            break
