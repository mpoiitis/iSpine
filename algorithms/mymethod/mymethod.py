import csv
import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.metrics import clustering_metrics, clustering
from utils.utils import load_data_trunc
import tensorflow as tf
from .models import DAE, DVAE, AE, VAE, ClusterBooster
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from utils.utils import save_results, salt_and_pepper, largest_eigval_smoothing_filter, preprocess_adj


class CMetricsTraceCallback(tf.keras.callbacks.Callback):
    def __init__(self, upd, best_cl, Cluster, feature, gnd):
        super(CMetricsTraceCallback, self).__init__()
        self.upd = upd
        self.best_db = best_cl
        self.Cluster = Cluster
        self.feature = feature
        self.gnd = gnd
        self.best_epoch = 0
        self.cluster_centers = self.Cluster.cluster_centers_

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % self.upd == 0:
            embeds = self.model.embed(self.feature)
            db, acc, nmi, f1, adjscore, centers = clustering(self.Cluster, embeds, self.gnd)
            tqdm.write("DB: {} ACC: {} NMI: {} ARI: {}".format(db, acc, nmi, adjscore))
            if db >= self.best_db:
                self.best_db = db
                self.best_acc = acc
                self.best_nmi = nmi
                self.best_f1 = f1
                self.best_ari = adjscore
                self.best_epoch = epoch
                self.cluster_centers = centers


def mymethod(args, feature, X, gnd):
    save_location = 'output/{}_{}_{}_power{}_epochs{}_hidden{}_dimension{}-batch{}-lr{}-drop{}'.format(args.input, args.method,
        args.model, args.power, args.epochs, args.hidden, args.dimension, args.batch_size, args.learning_rate, args.dropout)

    not_exists = True if not os.path.exists(save_location) else False # used to write metrics when the model is trained
    # TRAIN WITHOUT CLUSTER LABELS
    model, centers, acc, nmi, f1, ari = train(args, feature, X, gnd, save_location)

    # # TRAIN WITH CLUSTER LABELS ITERATIVELY
    # model, embeds, predict_labels, acc, nmi, f1, ari = retrain(args, feature, X, model, centers, ari, gnd)

    if args.save:
        if not_exists:
            write_results(args, acc, nmi, f1, ari)

        # save embeddings
        embeds = model.embed(feature)
        embeds_to_save = [e.numpy() for e in embeds]
        save_results(args, save_location, embeds_to_save)

    # tsne(embeds, gnd, args)


def run_mymethod(args):
    # import tensorflow as tf
    # with tf.device('/cpu:0'):
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)

    adj, feature, gnd, idx_train, idx_val, idx_test = load_data_trunc(args.input)

    # convert one hot labels to integer ones
    if args.input != "wiki":
        gnd = np.argmax(gnd, axis=1)
        feature = feature.todense()
    feature = feature.astype(np.float32)

    adj = sp.coo_matrix(adj)
    adj_normalized = preprocess_adj(adj)

    h = largest_eigval_smoothing_filter(adj_normalized)
    h_k = h ** args.power
    X = h_k.dot(feature)

    for _ in range(args.repeats):
        mymethod(args, feature, X, gnd)


def train(args, feature, X, gnd, save_location):
    """
    Model training
    :param args: cli arguments
    :param feature: original feature matrix
    :param X: the smoothed matrix
    :param gnd: ground truth labels
    """
    m = len(np.unique(gnd)) # number of clusters according to ground truth
    Cluster = KMeans(n_clusters=m)

    # cluster for initial values
    db, _, _, _, _, _ = clustering(Cluster, X, gnd)
    best_cl = db

    es = EarlyStopping(monitor='loss', patience=args.early_stopping)
    optimizer = Adam(lr=args.learning_rate)
    cc = CMetricsTraceCallback(args.upd, best_cl, Cluster, feature, gnd)

    # input is the plain feature matrix and output is the k-order convoluted. The model reconstructs the convolution!
    print('Training model for {}-order convolution'.format(args.power))
    if args.model == 'ae':
        model = AE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=optimizer, loss=MeanSquaredError())
        if os.path.exists(save_location):
            print('Model already exists. Loading it...')
            model.load_weights(save_location + '/checkpoint')
        else:
            print('Training model for {}-order convolution...'.format(args.power))
            model.fit(x=feature, y=X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, cc], verbose=1)
    elif args.model == 'vae':
        model = VAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=optimizer)
        if os.path.exists(save_location):
            print('Model already exists. Loading it...')
            model.load_weights(save_location + '/checkpoint')
        else:
            print('Training model for {}-order convolution...'.format(args.power))
            model.fit(feature, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, cc], verbose=1)
    elif args.model == 'dvae':
        # distort input features for denoising auto encoder
        distorted = salt_and_pepper(feature, 0.2)

        model = DVAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1],
                     dropout=args.dropout)
        model.compile(optimizer=optimizer)
        if os.path.exists(save_location):
            print('Model already exists. Loading it...')
            model.load_weights(save_location + '/checkpoint')
        else:
            print('Training model for {}-order convolution...'.format(args.power))
            model.fit(distorted, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, cc], verbose=1)
    elif args.model == 'dae':
        # distort input features for denoising auto encoder
        distorted = salt_and_pepper(feature, 0.2)

        model = DAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1],
                    dropout=args.dropout)
        model.compile(optimizer=optimizer)
        if os.path.exists(save_location):
            print('Model already exists. Loading it...')
            model.load_weights(save_location + '/checkpoint')
        else:
            print('Training model for {}-order convolution...'.format(args.power))
            model.fit(distorted, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, cc], verbose=1)
    else:
        return

    if not os.path.exists(save_location) and args.save:
        os.makedirs(save_location)
        model.save_weights(save_location + '/checkpoint')

    if not os.path.exists(save_location):
        tqdm.write("Optimization Finished!")
        tqdm.write(
            "BestDB: {} BestACC: {} BestNMI: {} BestARI: {}  Epoch: {}".format(cc.best_db, cc.best_acc, cc.best_nmi,
                                                                               cc.best_ari, cc.best_epoch))
        best_acc = cc.best_acc
        best_nmi = cc.best_nmi
        best_f1 = cc.best_f1
        best_ari = cc.best_ari
        best_centers = cc.cluster_centers

    return model, best_centers, best_acc, best_nmi, best_f1, best_ari


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

