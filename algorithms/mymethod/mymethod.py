import scipy.io as sio
import csv
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from textwrap import wrap
from sklearn.cluster import KMeans
from utils.metrics import clustering_metrics, square_dist
from .models import DAE, DVAE, AE, VAE, ClusterBooster
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.utils import save_results, salt_and_pepper
import matplotlib.pyplot as plt


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


def mymethod(args):
    import tensorflow as tf
    with tf.device('/cpu:0'):
        if not os.path.exists('output/mymethod'):
            os.makedirs('output/mymethod')

        dataset = args.input
        data = sio.loadmat('data/agc_data/{}.mat'.format(dataset))
        feature = data['fea']
        feature = feature.astype(np.float32)
        if sp.issparse(feature):
            feature = feature.todense()

        adj = data['W']
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - 1
        gnd = gnd[0, :]

        num_nodes = adj.shape[0]
        m = len(np.unique(gnd))
        adj = sp.coo_matrix(adj)

        adj_normalized = preprocess_adj(adj)
        adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2

        adj_normalized_k = adj_normalized ** args.power
        X = adj_normalized_k.dot(feature)

        # feed k-order convolution to autoencoder

        es = EarlyStopping(monitor='loss', patience=args.early_stopping)
        optimizer = Adam(lr=args.learning_rate)

        # TRAIN WITHOUT CLUSTER LABELS
        # input is the plain feature matrix and output is the k-order convoluted. The model reconstructs the convolution!
        print('Training model for {}-order convolution'.format(args.power))
        if args.model == 'ae':
            model = AE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
            model.compile(optimizer=optimizer, loss=MeanSquaredError())
            model.fit(feature, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
        elif args.model == 'vae':
            model = VAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
            model.compile(optimizer=optimizer)
            model.fit(feature, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
        elif args.model == 'dvae':
            # distort input features for denoising auto encoder
            distorted = salt_and_pepper(feature, 0.2)

            model = DVAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1],
                        dropout=args.dropout)
            model.compile(optimizer=optimizer)
            model.fit(distorted, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
        elif args.model == 'dae':
            # distort input features for denoising auto encoder
            distorted = salt_and_pepper(feature, 0.2)

            model = DAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1],
                        dropout=args.dropout)
            model.compile(optimizer=optimizer)
            model.fit(distorted, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)

        embeds = model.embed(feature)

        # predict cluster assignments according to the inital autoencoder
        kmeans = KMeans(n_clusters=m).fit(embeds)
        predict_labels = kmeans.predict(embeds)

        metric = square_dist(predict_labels, feature)
        cm = clustering_metrics(gnd, predict_labels)
        ac, nm, f1 = cm.evaluationClusterModelFromLabel()

        print('Power: {}'.format(args.power), 'Iteration: 0  intra_dist: {}'.format(metric), 'acc: {}'.format(ac),
              'nmi: {}'.format(nm),
              'f1: {}'.format(f1))

        # # TRAIN WITH CLUSTER LABELS ITERATIVELY
        # model, embeds, predict_labels, ac, nm, f1, metric = retrain(args, feature, X, model, kmeans.cluster_centers_, metric, gnd)

        # write_results(args, ac, nm, f1, metric)

        if args.save:
            # save embeddings
            embeds = model.embed(feature)
            embeds_to_save = [e.numpy() for e in embeds]
            save_results(args, embeds_to_save)

        # tsne(embeds, gnd, predict_labels, args)

def run_mymethod(args):
    for _ in range(args.repeats):
        mymethod(args)


def retrain(args, feature, X, model, centers, metric, gnd):
    """
    Self-supervision
    :param args: cli arguments
    :param feature: original feature matrix
    :param X: the smoothed matrix
    :param model: the pretrained model
    :param centers: cluster centers using the embeddings of the pretrained model
    :param metric: metric to optimize
    :param gnd: ground truth labels
    """
    es = EarlyStopping(monitor='loss', patience=args.early_stopping)
    optimizer = Adam(lr=args.learning_rate)

    iteration = 0
    if args.c_epochs != 0:
        while True:
            # repeat cluster boosting as long as the model is getting better with respect to intra cluster distance
            iteration += 1

            model = ClusterBooster(model, centers)
            model.compile(optimizer=optimizer)
            if args.model == 'ae' or args.model == 'vae':
                model.fit(feature, X, epochs=args.c_epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es],
                          verbose=0)
            else:  # dae or dvae
                distorted = salt_and_pepper(feature, 0.2)
                model.fit(distorted, X, epochs=args.c_epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es],
                          verbose=0)

            embeds = model.embed(feature)

            kmeans = KMeans(n_clusters=len(centers)).fit(embeds)
            predict_labels = kmeans.predict(embeds)
            new_dist = square_dist(predict_labels, feature)

            if new_dist > metric:
                break
            else:
                metric = new_dist

            cm = clustering_metrics(gnd, predict_labels)
            ac, nm, f1 = cm.evaluationClusterModelFromLabel()
            print('Power: {}'.format(args.power), 'Iteration: {}'.format(iteration), '   intra_dist: {}'.format(metric),
                  'acc: {}'.format(ac),
                  'nmi: {}'.format(nm),
                  'f1: {}'.format(f1))

    return model, embeds, predict_labels, ac, nm, f1, metric


def write_results(args, ac, nm, f1, metric):
    file_exists = os.path.isfile('output/mymethod/results.csv')
    with open('output/mymethod/results.csv', 'a') as f:
        columns = ['Dataset', 'Model', 'Dimension', 'Hidden', 'Epochs', 'Batch Size', 'Learning Rate', 'Dropout',
                   'Cluster Epochs', 'Power', 'Accuracy', 'NMI', 'F1', 'Intra Distance']
        writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=columns)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow({'Dataset': args.input, 'Model': args.model, 'Dimension': args.dimension, 'Hidden': args.hidden,
                         'Epochs': args.epochs, 'Batch Size': args.batch_size, 'Learning Rate': args.learning_rate,
                         'Dropout': args.dropout, 'Cluster Epochs': args.c_epochs,
                         'Power': args.power, 'Accuracy': ac, 'NMI': nm, 'F1': f1, 'Intra Distance': metric})


def tsne(embeds, gnd, predict_labels, args):
    from sklearn.manifold import TSNE
    import seaborn as sns
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("bright", len(np.unique(gnd)))
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(embeds)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], ax=ax1, hue=gnd, legend='full', palette=palette)
    ax1.set_title('T-SNE {} ground truth'.format(args.input))
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], ax=ax2, hue=predict_labels, legend='full', palette=palette)
    ax2.set_title('T-SNE {} predicted labels'.format(args.input))
    plt.savefig('figures/mymethod/tsne/{}_{}epochs_{}dims_{}hidden.png'.format(args.input, args.epochs, args.dimension,
                                                                               args.hidden), format='png')
    plt.show()


def plot_results(config, pivot='Learning Rate'):

    filepath = 'figures/mymethod/{}'.format(config['Model'])
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    data = pd.read_csv('output/mymethod/results.csv')
    for k, v in config.items():
        data = data.loc[data[k] == v]

    data = data.sort_values(pivot)

    unique_xaxis = np.unique(data[pivot])

    if config['Dataset'] == 'cora':
        agc_acc = len(unique_xaxis) * [0.6892]
        agc_f1 = len(unique_xaxis) * [0.6561]
        agc_nmi = len(unique_xaxis) * [0.5368]
    elif config['Dataset'] == 'citeseer':
        agc_acc = len(unique_xaxis) * [0.6700]
        agc_f1 = len(unique_xaxis) * [0.6248]
        agc_nmi = len(unique_xaxis) * [0.4113]
    elif config['Dataset'] == 'pubmed':
        agc_acc = len(unique_xaxis) * [0.6978]
        agc_f1 = len(unique_xaxis) * [0.6872]
        agc_nmi = len(unique_xaxis) * [0.3159]
    else:
        return

    acc_means = data.groupby(pivot, as_index=False)['Accuracy'].mean()
    nmi_means = data.groupby(pivot, as_index=False)['NMI'].mean()
    f1s_means = data.groupby(pivot, as_index=False)['F1'].mean()

    acc_stds = data.groupby(pivot, as_index=False)['Accuracy'].std()
    nmi_stds = data.groupby(pivot, as_index=False)['NMI'].std()
    f1_stds = data.groupby(pivot, as_index=False)['F1'].std()

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(acc_means[pivot], acc_means['Accuracy'], color='purple', label='Acc', marker='x')
    ax[0].fill_between(acc_means[pivot], acc_means['Accuracy'] - acc_stds['Accuracy'], acc_means['Accuracy'] + acc_stds['Accuracy'], color='purple', alpha=0.2)
    ax[0].plot(acc_means[pivot], agc_acc, color='black', label='AGC')
    ax[0].legend()

    ax[1].plot(f1s_means[pivot], f1s_means['F1'], color='yellow', label='F1', marker='x')
    ax[1].fill_between(f1s_means[pivot], f1s_means['F1'] - f1_stds['F1'], f1s_means['F1'] + f1_stds['F1'], color='yellow', alpha=0.2)
    ax[1].plot(f1s_means[pivot], agc_f1, color='black', label='AGC')
    ax[1].legend()

    ax[2].plot(nmi_means[pivot], nmi_means['NMI'], color='green', label='NMI', marker='x')
    ax[2].fill_between(nmi_means[pivot], nmi_means['NMI'] - nmi_stds['NMI'], nmi_means['NMI'] + nmi_stds['NMI'], color='green', alpha=0.2)
    ax[2].plot(nmi_means[pivot], agc_nmi, color='black', label='AGC')
    ax[2].legend()

    ax[0].set_xlabel(pivot)
    ax[1].set_xlabel(pivot)
    ax[2].set_xlabel(pivot)
    ax[0].set_ylabel('Score')
    ax[1].set_ylabel('Score')
    ax[2].set_ylabel('Score')

    num_items = len(config.keys())
    title = ''
    for i, (k, v) in enumerate(config.items()):
        if i == num_items - 1:
            title += k + ':' + str(v)
        else:
            title += k + ':' + str(v) + ', '
    plt.suptitle("\n".join(wrap(title, 75)))

    filepath = filepath + '/'
    for i, v in enumerate(config.values()):
        if i == num_items - 1:
            filepath += str(v)
        else:
            filepath += str(v) + '_'
    filepath += '.png'
    plt.savefig(filepath, format='png')
    plt.show()
