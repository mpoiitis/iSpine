import csv
import os
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from utils.metrics import clustering_metrics
from utils.utils import load_data_trunc
from utils.plots import tsne
from .models import DAE, DVAE, AE, VAE, ClusterBooster
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.utils import save_results, salt_and_pepper, largest_eigval_smoothing_filter
from scipy.sparse.linalg.eigen.arpack import eigsh


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


def mymethod(args, feature, X, gnd):

    # TRAIN WITHOUT CLUSTER LABELS
    model, centers, ac, nm, f1, ari = train(args, feature, X, gnd)

    # # TRAIN WITH CLUSTER LABELS ITERATIVELY
    # model, embeds, predict_labels, ac, nm, f1, ari = retrain(args, feature, X, model, centers, ari, gnd)

    if args.save:
        write_results(args, ac, nm, f1, ari)

        # save embeddings
        embeds = model.embed(feature)
        embeds_to_save = [e.numpy() for e in embeds]
        save_results(args, embeds_to_save)

    # tsne(embeds, gnd, args)


def run_mymethod(args):
    # import tensorflow as tf
    # with tf.device('/cpu:0'):
    if not os.path.exists('output/mymethod'):
        os.makedirs('output/mymethod')

    adj, feature, gnd, idx_train, idx_val, idx_test = load_data_trunc(args.input)

    if args.input != "wiki":
        gnd = np.argmax(gnd, axis=1)  # convert one hot labels to integer ones
        feature = feature.todense()
    feature = feature.astype(np.float32)

    # # find best convolution order according to eigenvalue
    # eigvals, _ = np.linalg.eig(adj)
    # sum = np.sum(eigvals)
    # partial = 0
    # best_power = -1
    # for idx, eigval in enumerate(eigvals):
    #     partial += eigval
    #     if (partial / sum) >= 0.9:  # keep the eigenvalues corresponding to 90% of the matrix info
    #         best_power = idx + 1
    #         break
    # print(best_power)

    h = largest_eigval_smoothing_filter(adj)
    h_k = h ** args.power

    X = h_k.dot(feature)

    for _ in range(args.repeats):
        mymethod(args, feature, X, gnd)


def train(args, feature, X, gnd):
    """
    Model training
    :param args: cli arguments
    :param feature: original feature matrix
    :param X: the smoothed matrix
    :param gnd: ground truth labels
    """
    m = len(np.unique(gnd)) # number of clusters according to ground truth

    es = EarlyStopping(monitor='loss', patience=args.early_stopping)
    optimizer = Adam(lr=args.learning_rate)
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

    # metric = square_dist(predict_labels, feature)
    cm = clustering_metrics(gnd, predict_labels)
    ac, nm, f1, ari = cm.evaluationClusterModelFromLabel()

    print('Power: {}'.format(args.power), 'Iteration: 0  ari: {}'.format(ari), 'acc: {}'.format(ac),
          'nmi: {}'.format(nm),
          'f1: {}'.format(f1))
    return model, kmeans.cluster_centers_, ac, nm, f1, ari


def retrain(args, feature, X, model, centers, ari, gnd):
    """
    Self-supervision
    :param args: cli arguments
    :param feature: original feature matrix
    :param X: the smoothed matrix
    :param model: the pretrained model
    :param centers: cluster centers using the embeddings of the pretrained model
    :param ari: adjusted rand index. Metric to optimize
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
            # new_dist = square_dist(predict_labels, feature)

            cm = clustering_metrics(gnd, predict_labels)
            ac, nm, f1, new_ari = cm.evaluationClusterModelFromLabel()

            if new_ari < ari:
                break
            else:
                ari = new_ari

            print('Power: {}'.format(args.power), 'Iteration: {}'.format(iteration), '   ari: {}'.format(new_ari),
                  'acc: {}'.format(ac),
                  'nmi: {}'.format(nm),
                  'f1: {}'.format(f1))

    return model, embeds, predict_labels, ac, nm, f1, new_ari


def write_results(args, ac, nm, f1, ari):
    file_exists = os.path.isfile('output/mymethod/results.csv')
    with open('output/mymethod/results.csv', 'a') as f:
        columns = ['Dataset', 'Model', 'Dimension', 'Hidden', 'Epochs', 'Batch Size', 'Learning Rate', 'Dropout',
                   'Cluster Epochs', 'Power', 'Accuracy', 'NMI', 'F1', 'ARI']
        writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=columns)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow({'Dataset': args.input, 'Model': args.model, 'Dimension': args.dimension, 'Hidden': args.hidden,
                         'Epochs': args.epochs, 'Batch Size': args.batch_size, 'Learning Rate': args.learning_rate,
                         'Dropout': args.dropout, 'Cluster Epochs': args.c_epochs,
                         'Power': args.power, 'Accuracy': ac, 'NMI': nm, 'F1': f1, 'ARI': ari})

