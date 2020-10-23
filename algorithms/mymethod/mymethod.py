import scipy.io as sio
import time
import os
import numpy as np
import scipy.sparse as sp
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


def run_mymethod(args):

    if not os.path.exists('output/tmp'):
        os.makedirs('output/tmp')
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

    exp_accs = []
    exp_nmis = []
    exp_f1s = []
    # repeat the experiment 10 times to check the stableness of the model
    for i in range(10):
        acc_list = []
        nmi_list = []
        f1_list = []
        powers = []
        for power in range(1, 8):

            # trans_matrix = adj_normalized / adj_normalized.sum(axis=0)
            # M = trans_matrix
            # if power > 1:
            #     for i in range(2, power):
            #         M += np.power(trans_matrix, i)
            #     M /= power
            # X = M.dot(feature)

            adj_normalized_k = adj_normalized ** power
            X = adj_normalized_k.dot(feature)
            # K = X.dot(X.transpose())
            # W = 1/2 * (np.absolute(K) + np.absolute(K.transpose()))
            W = X
            # feed k-order convolution to autoencoder

            es = EarlyStopping(monitor='loss', patience=args.early_stopping)
            optimizer = Adam(lr=args.learning_rate)
            # # save weights
            # checkpoint_filepath = 'output/tmp/checkpoint'
            # model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='loss',
            #     mode='min', save_best_only=True)

            # TRAIN WITHOUT CLUSTER LABELS
            # input is the plain feature matrix and output is the k-order convoluted. The model reconstructs the convolution!
            print('Training model for {}-order convolution'.format(power))
            if args.model == 'ae':
                model = AE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=W.shape[1], dropout=args.dropout)
                model.compile(optimizer=optimizer, loss=MeanSquaredError())
                model.fit(feature, W, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
                # model.fit(u, u, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, model_checkpoint_callback], verbose=0)
            elif args.model == 'vae':
                model = VAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=W.shape[1], dropout=args.dropout)
                model.compile(optimizer=optimizer)
                model.fit(feature, W, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
                # model.fit(u, u, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, model_checkpoint_callback], verbose=0)
            elif args.model == 'dvae':
                # distort input features for denoising auto encoder
                distorted = salt_and_pepper(feature, 0.2)

                model = DVAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=W.shape[1],
                            dropout=args.dropout)
                model.compile(optimizer=optimizer)
                model.fit(distorted, W, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
                # model.fit(u_distorted, u, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, model_checkpoint_callback], verbose=0)
            elif args.model == 'dae':
                # distort input features for denoising auto encoder
                distorted = salt_and_pepper(feature, 0.2)

                model = DAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=W.shape[1],
                            dropout=args.dropout)
                model.compile(optimizer=optimizer)
                model.fit(distorted, W, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
                # model.fit(u_distorted, u, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, model_checkpoint_callback], verbose=0)
            # model.load_weights(checkpoint_filepath)
            embeds = model.embed(feature)

            if args.save and power == args.save:
                # save embeddings
                embeds_to_save = [e.numpy() for e in embeds]
                save_results(args, embeds_to_save)

            # predict cluster assignments according to the inital autoencoder
            kmeans = KMeans(n_clusters=m).fit(embeds)
            predict_labels = kmeans.predict(embeds)

            intraD = square_dist(predict_labels, feature)
            cm = clustering_metrics(gnd, predict_labels)
            ac, nm, f1 = cm.evaluationClusterModelFromLabel()
            iteration = 0
            print('Power: {}'.format(power), 'Iteration: {}'.format(iteration),  '  intra_dist: {}'.format(intraD), 'acc: {}'.format(ac),
                  'nmi: {}'.format(nm),
                  'f1: {}'.format(f1))

            # # TRAIN WITH CLUSTER LABELS ITERATIVELY
            # while True:
            #     # repeat cluster boosting as long as the model is getting better with respect to intra cluster distance
            #     iteration += 1
            #
            #     model = ClusterBooster(model, kmeans.cluster_centers_)
            #     model.compile(optimizer=optimizer)
            #     if args.model == 'ae' or args.model == 'vae':
            #         model.fit(feature, W, epochs=args.c_epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
            #     else: #dae or dvae
            #         model.fit(distorted, W, epochs=args.c_epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
            #
            #     embeds = model.embed(feature)
            #
            #     kmeans = KMeans(n_clusters=m).fit(embeds)
            #     predict_labels = kmeans.predict(embeds)
            #     new_dist = square_dist(predict_labels, feature)
            #
            #     if new_dist > intraD:
            #         break
            #     else:
            #         intraD = new_dist
            #
            #     cm = clustering_metrics(gnd, predict_labels)
            #     ac, nm, f1 = cm.evaluationClusterModelFromLabel()
            #     print('Power: {}'.format(power), 'Iteration: {}'.format(iteration), '   intra_dist: {}'.format(intraD), 'acc: {}'.format(ac),
            #           'nmi: {}'.format(nm),
            #           'f1: {}'.format(f1))

            powers.append(power)
            acc_list.append(ac)
            nmi_list.append(nm)
            f1_list.append(f1)


        best_index = np.argmax(f1_list)
        best_power = powers[best_index]
        print('Best power: {}'.format(best_power))

        exp_accs.append(acc_list)
        exp_nmis.append(nmi_list)
        exp_f1s.append(f1_list)

    plot_results(powers, exp_accs, exp_nmis, exp_f1s, args)



def plot_results(powers, exp_accs, exp_nmis, exp_f1s, args):

    filepath = 'figures/{}/{}'.format(args.method, args.model)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if args.input == 'cora':
        agc_acc = len(powers) * [0.6892]
        agc_f1 = len(powers) * [0.6561]
        agc_nmi = len(powers) * [0.5368]
    elif args.input == 'citeseer':
        agc_acc = len(powers) * [0.6700]
        agc_f1 = len(powers) * [0.6248]
        agc_nmi = len(powers) * [0.4113]
    elif args.input == 'pubmed':
        agc_acc = len(powers) * [0.6978]
        agc_f1 = len(powers) * [0.6872]
        agc_nmi = len(powers) * [0.3159]
    else:
        return

    exp_accs = np.array(exp_accs)
    exp_nmis = np.array(exp_nmis)
    exp_f1s = np.array(exp_f1s)

    # mean across the columns, meaning for each k
    acc_mean = np.mean(exp_accs, axis=0)
    nmi_mean = np.mean(exp_nmis, axis=0)
    f1_mean = np.mean(exp_f1s, axis=0)

    #std across the columns, meaning for each k
    acc_std = np.std(exp_accs, axis=0)
    nmi_std = np.std(exp_nmis, axis=0)
    f1_std = np.std(exp_f1s, axis=0)

    for i in range(len(powers)):
        print('Power: {}    acc:{} +- {}    nmi:{} +- {}   f1:{} +- {}'.format(powers[i], acc_mean[i], acc_std[i], nmi_mean[i], nmi_std[i], f1_mean[i], f1_std[i]))

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(powers, acc_mean, color='purple', label='Acc')
    ax[0].fill_between(powers, acc_mean + acc_std, acc_mean - acc_std, facecolor='purple', alpha=0.2)
    ax[0].plot(powers, agc_acc, color='black', label='AGC')
    ax[0].legend()

    ax[1].plot(powers, f1_mean, color='yellow', label='F1')
    ax[1].fill_between(powers, f1_mean + f1_std, f1_mean - f1_std, facecolor='yellow', alpha=0.2)
    ax[1].plot(powers, agc_f1, color='black', label='AGC')
    ax[1].legend()

    ax[2].plot(powers, nmi_mean, color='green', label='NMI')
    ax[2].fill_between(powers, nmi_mean + nmi_std, nmi_mean - nmi_std, facecolor='green', alpha=0.2)
    ax[2].plot(powers, agc_nmi, color='black', label='AGC')
    ax[2].legend()

    ax[0].set_xlabel('k')
    ax[1].set_xlabel('k')
    ax[2].set_xlabel('k')
    ax[0].set_ylabel('Score')
    ax[1].set_ylabel('Score')
    ax[2].set_ylabel('Score')
    plt.suptitle('dataset:{}, learning rate: {}, epochs: {}, c-epochs: {}, dimension: {}, dropout: {}'.format(args.input, args.learning_rate, args.epochs, args.c_epochs, args.dimension, args.dropout))

    filepath = filepath + '/' + str(args.input) + '_' + str(args.model) + '_' + str(args.epochs) + '_' + str(args.c_epochs) + '_' + str(args.dimension) + '_' + str(args.learning_rate) + '_' + str(args.dropout) + '.png'
    plt.savefig(filepath, format='png')
    plt.show()