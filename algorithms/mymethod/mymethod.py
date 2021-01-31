import csv
import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.metrics import clustering
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

        self.best_acc = -1
        self.best_nmi = -1
        self.best_f1 = -1
        self.best_ari = -1
        self.best_epoch = -1
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

    # CREATE MODEL
    if args.model == 'ae':
        model = AE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=Adam(lr=args.learning_rate), loss=MeanSquaredError())
    elif args.model == 'vae':
        model = VAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=Adam(lr=args.learning_rate))
    elif args.model == 'dae':
        model = DAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=Adam(lr=args.learning_rate))
    else:
        model = DVAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=Adam(lr=args.learning_rate))

    # TRAINING OR LOAD MODEL IF IT EXISTS
    if not os.path.exists(save_location):
        # TRAINING
        model, centers, acc, nmi, f1, ari = train(args, feature, X, gnd, model)

        # SELF-SUPERVISION
        if args.c_epochs:
            model, acc, nmi, f1, ari = self_supervise(args, feature, X, gnd, model, centers)
        if args.save:
            os.makedirs(save_location)
            model.save_weights(save_location + '/checkpoint')
            write_results(args, acc, nmi, f1, ari)
    else:
        print('Model already exists. Loading it...')
        model.load_weights(save_location + '/checkpoint')

    # SAVE EMBEDDINGS
    if args.save:
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


def train(args, feature, X, gnd, model):
    """
    Model training
    :param args: cli arguments
    :param feature: original feature matrix
    :param X: the smoothed matrix
    :param gnd: ground truth labels
    :param model: the nn to be trained
    """

    # INITIAL CLUSTERING
    m = len(np.unique(gnd)) # number of clusters according to ground truth
    Cluster = KMeans(n_clusters=m)
    db, _, _, _, _, _ = clustering(Cluster, X, gnd)
    best_cl = db

    # ADD NOISE IN CASE OF DENOISING MODELS
    if args.model == 'ae' or args.model == 'vae':
        input = feature
    else:
        input = salt_and_pepper(feature, 0.2)

    # CALLBACKS
    es = EarlyStopping(monitor='loss', patience=args.early_stopping)
    cc = CMetricsTraceCallback(args.upd, best_cl, Cluster, input, gnd)

    # TRAINING
    # input is the plain feature matrix and output is the k-order convoluted. The model reconstructs the convolution!
    print('Training model for {}-order convolution'.format(args.power))
    model.fit(input, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, cc], verbose=1)

    # REPORTING
    tqdm.write("Optimization Finished!")
    tqdm.write("BestDB: {} BestACC: {} BestNMI: {} BestARI: {}  Epoch: {}".format(cc.best_db, cc.best_acc, cc.best_nmi,
                                                                           cc.best_ari, cc.best_epoch))
    best_acc = cc.best_acc
    best_nmi = cc.best_nmi
    best_f1 = cc.best_f1
    best_ari = cc.best_ari
    best_centers = cc.cluster_centers

    if cc.best_epoch == -1:
        raise Exception('Training did not improve clustering. Aborting method.')

    return model, best_centers, best_acc, best_nmi, best_f1, best_ari


def self_supervise(args, feature, X, gnd, model, centers):
    """
    Self-supervision using KL loss
    :param args: cli arguments
    :param feature: original feature matrix
    :param X: the smoothed matrix
    :param gnd: ground truth labels
    :param model: the pretrained model
    :param centers: cluster centers using the embeddings of the pretrained model
    """
    # INITIAL CLUSTERING
    m = len(np.unique(gnd))  # number of clusters according to ground truth
    Cluster = KMeans(n_clusters=m)
    db, _, _, _, _, _ = clustering(Cluster, X, gnd)
    best_cl = db

    # ADD NOISE IN CASE OF DENOISING MODELS
    if args.model == 'ae' or args.model == 'vae':
        input = feature
    else:
        input = salt_and_pepper(feature, 0.2)

    # CALLBACKS
    es = EarlyStopping(monitor='loss', patience=args.early_stopping)
    cc = CMetricsTraceCallback(args.upd, best_cl, Cluster, input, gnd)


    # CREATE MODEL
    model = ClusterBooster(model, centers)
    model.compile(optimizer=Adam(lr=args.learning_rate))

    # TRAIN
    print('Self-supervision started')
    model.fit(input, epochs=args.c_epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, cc], verbose=1)

    # REPORTING
    tqdm.write("Optimization Finished!")
    tqdm.write(
        "BestDB: {} BestACC: {} BestNMI: {} BestARI: {}  Epoch: {}".format(cc.best_db, cc.best_acc, cc.best_nmi,
                                                                           cc.best_ari, cc.best_epoch))
    best_acc = cc.best_acc
    best_nmi = cc.best_nmi
    best_f1 = cc.best_f1
    best_ari = cc.best_ari

    return model, best_acc, best_nmi, best_f1, best_ari


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