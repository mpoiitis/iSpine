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

    acc_list = []
    nmi_list = []
    f1_list = []
    powers = []
    d_intra = []

    t = time.time()
    adj_normalized = preprocess_adj(adj)
    adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2


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
        embeds = model.embed(W)

        if args.save and power == args.save:
            # save embeddings
            embeds_to_save = [e.numpy() for e in embeds]
            save_results(args, embeds_to_save)

        # predict cluster assignments according to the inital autoencoder
        kmeans = KMeans(n_clusters=m).fit(embeds)
        predict_labels = kmeans.predict(embeds)

        # TRAIN WITH CLUSTER LABELS ITERATIVELY
        intraD = square_dist(predict_labels, X)
        cm = clustering_metrics(gnd, predict_labels)
        ac, nm, f1 = cm.evaluationClusterModelFromLabel()
        iteration = 0
        print('Power: {}'.format(power), 'Iteration: {}'.format(iteration),  '  intra_dist: {}'.format(intraD), 'acc: {}'.format(ac),
              'nmi: {}'.format(nm),
              'f1: {}'.format(f1))
        while True:
            # repeat cluster boosting as long as the model is getting better
            iteration += 1

            model = ClusterBooster(model, kmeans.cluster_centers_)
            model.compile(optimizer=optimizer)
            if args.model == 'ae' or args.model == 'vae':
                model.fit(feature, W, epochs=args.c_epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)
            else: #dae or dvae
                model.fit(distorted, W, epochs=args.c_epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=0)

            embeds = model.embed(W)

            kmeans = KMeans(n_clusters=m).fit(embeds)
            predict_labels = kmeans.predict(embeds)
            new_dist = square_dist(predict_labels, W)

            if new_dist > intraD:
                break
            else:
                intraD = new_dist

            cm = clustering_metrics(gnd, predict_labels)
            ac, nm, f1 = cm.evaluationClusterModelFromLabel()
            print('Power: {}'.format(power), 'Iteration: {}'.format(iteration), '   intra_dist: {}'.format(intraD), 'acc: {}'.format(ac),
                  'nmi: {}'.format(nm),
                  'f1: {}'.format(f1))

        powers.append(power)
        acc_list.append(ac)
        nmi_list.append(nm)
        f1_list.append(f1)

    best_index = np.argmax(f1_list)
    best_power = powers[best_index]
    print('Best power: {}'.format(best_power))

    # plot_results(d_intra, acc_list, nmi_list, f1_list, powers, args)



def plot_results(intra, acc, nmi, f1, powers, args):

    filepath = 'figures/{}/{}'.format(args.method, args.model)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    intra[0] = - 7

    powers = np.array(powers)
    acc = np.array(acc)
    f1 = np.array(f1)
    nmi = np.array(nmi)
    intra = np.array(intra)

    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(powers, acc, color='purple', label='Acc')
    axs[0].plot(powers, f1, color='yellow', label='F1')
    axs[0].plot(powers, nmi, color='green', label='NMI')
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('Score')
    axs[1].plot(powers, intra, color='blue', label='D_Intra')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('Score')
    fig.suptitle('learning rate: {}, epochs: {}, dimension: {}, dropout: {}'.format(args.learning_rate, args.epochs, args.dimension, args.dropout))
    axs[0].legend()
    axs[1].legend()
    filepath = filepath + '/' + str(args.model) + '_' + str(args.epochs) + '_' + str(args.dimension) + '_' + str(args.learning_rate) + '_' + str(args.dropout) + '.png'
    plt.savefig(filepath, format='png')
    plt.show()