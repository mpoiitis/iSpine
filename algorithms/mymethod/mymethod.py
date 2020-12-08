import csv
import os
import time
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import normalize
from sklearn import metrics
from utils.metrics import clustering_metrics
from utils.utils import load_data_trunc
from .models import AE, ClusterBooster
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from utils.utils import save_results, salt_and_pepper, largest_eigval_smoothing_filter


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


def update_similarity(z, upper_threshold, lower_treshold):
    """
    Calculate the pairwise similarity matrix.
    Rank node pairs, and get a sample accoring to the given thresholds
    """
    cosine = np.matmul(z, np.transpose(z))
    cosine = cosine.reshape([-1, ])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)


def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth


def clustering(Cluster, feature, true_labels):
    """
    Clusters the given features and reports the corresponding metrics.
    Davies-Bouldin index is the metric used to decide the better clustering
    :param Cluster: The clustering algorithm. Default is spectral clustering
    :param feature: the feature matrix
    :param true_labels: ground truth labels
    """
    f_adj = np.matmul(feature, np.transpose(feature))
    predict_labels = Cluster.fit_predict(f_adj)

    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    acc, nmi, f1, ari = cm.evaluationClusterModelFromLabel()

    return db, acc, nmi, f1, ari


def run_mymethod(args):
    # import tensorflow as tf
    # with tf.device('/cpu:0'):
    if not os.path.exists('output/mymethod'):
        os.makedirs('output/mymethod')
    # READ INPUT
    adj, feature, labels, idx_train, idx_val, idx_test = load_data_trunc(args.input)

    if args.input != "wiki":
        labels = np.argmax(labels, axis=1)  # convert one hot labels to integer ones
        feature = feature.todense()
    feature = feature.astype(np.float32)
    n_nodes, feat_dim = feature.shape

    # SET CLUSTERING PARAMS
    n_clusters = len(np.unique(labels))
    Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)

    # LAPLACIAN SMOOTHING
    h = largest_eigval_smoothing_filter(adj)
    h_k = h ** args.power
    X = h_k.dot(feature)

    # INITIAL CLUSTERING
    best_db, best_acc, best_nmi, best_f1, best_ari = clustering(Cluster, feature, labels)

    # SET UP FOR MODEL TRAINING
    pos_num = len(adj.indices)
    neg_num = n_nodes * n_nodes - pos_num

    up_eta = (args.upth_ed - args.upth_st) / (args.epochs / args.update)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs / args.update)

    pos_inds, neg_inds = update_similarity(normalize(feature), args.upth_st, args.lowth_st)
    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)

    batch_size = min(args.batch_size, len(pos_inds))

    # MODEL
    model = AE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
    optimizer = Adam(lr=args.learning_rate)
    loss_fn = MeanSquaredError()

    # TRAINING
    for epoch in range(args.epochs):
        start, end = 0, batch_size
        batch_num = 0
        length = len(pos_inds)

        # BATCH TRAINING
        while (end <= length):
            negative_sample = np.random.choice(neg_inds, size=end - start) # get a sample from negative indices
            sample_indices = np.concatenate((pos_inds[start:end], negative_sample), 0) # create training by positive and negative samples
            sample_indices = sample_indices // n_nodes # the original array was flattened. We need to get back the original indices
            t = time.time()

            x_batch = feature[sample_indices, :]
            y_batch = X[sample_indices, :]
            with tf.GradientTape() as tape:
                logits = model(x_batch)
                loss_value = loss_fn(y_batch, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # update indices for batch sampling
            start = end
            batch_num += 1
            if end < length and end + batch_size >= length:
                end += length - end
            else:
                end += batch_size

        # EVALUATION THROUGH CLUSTERING
        if (epoch + 1) % args.update == 0:
            z = model.embed(feature)
            z_numpy = [e.numpy() for e in z]
            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(z_numpy, upth, lowth)
            batch_size = min(args.batch_size, len(pos_inds))

            tqdm.write("Epoch: {}, train_loss={:.5f}, time={:.5f}".format(epoch + 1, loss_value, time.time() - t))
            db, acc, nmi, f1, ari = clustering(Cluster, z_numpy, labels)
            tqdm.write("Davies-Bouldin={}  Acc={} NMI={} ARI={} \n".format(db, acc, nmi, ari))
            if db >= best_db:
                best_db = db
                best_acc = acc
                best_nmi = nmi
                best_f1 = f1
                best_ari = ari

                best_z = z_numpy

    tqdm.write("Optimization Finished!")
    tqdm.write('best_acc: {}, best_nmi: {}, best_ari: {}'.format(best_acc, best_nmi, best_ari))

    if args.save:
        write_results(args, best_acc, best_nmi, best_f1, best_ari)
        save_results(args, best_z)

    # tsne(embeds, gnd, args)


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

