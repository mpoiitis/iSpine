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
from tensorflow.keras.losses import MeanSquaredError, KLDivergence
from tensorflow.keras.callbacks import EarlyStopping
from utils.utils import save_results, salt_and_pepper, largest_eigval_smoothing_filter, preprocess_adj


def run_kspace_grid_search():
    from tensorboard.plugins.hparams import api as hp
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)

    adj, feature, gnd, idx_train, idx_val, idx_test = load_data_trunc("cora")
    gnd = np.argmax(gnd, axis=1)
    feature = feature.todense()
    feature = feature.astype(np.float32)
    adj = sp.coo_matrix(adj)
    adj_normalized = preprocess_adj(adj)

    h = largest_eigval_smoothing_filter(adj_normalized)

    HP_POWERS = hp.HParam('power', hp.Discrete([5, 6, 7, 8]))
    HP_BATCH = hp.HParam('batch', hp.Discrete([32, 64, 100, 200]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    session_num = 0
    for power in HP_POWERS.domain.values:
        for batch in HP_BATCH.domain.values:
            for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                for optimizer in HP_OPTIMIZER.domain.values:
                    hparams = {
                        HP_POWERS: power,
                        HP_BATCH: batch,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    logdir = 'logs/hparam_tuning/' + run_name

                    h_k = h ** hparams[HP_POWERS]
                    X = h_k.dot(feature)

                    model = AE(layers=2, dims=[200, 100], output_dim=X.shape[1], dropout=hparams[HP_DROPOUT])
                    model.compile(optimizer=hparams[HP_OPTIMIZER], loss=MeanSquaredError())
                    es = EarlyStopping(monitor='loss', patience=20)
                    model.fit(feature, X, epochs=500, batch_size=hparams[HP_BATCH], shuffle=True, callbacks=[es, tf.keras.callbacks.TensorBoard(logdir), hp.KerasCallback(logdir, hparams)], verbose=1)
                    mse = model.evaluate(feature, X)
                    tf.summary.scalar('mse', mse, step=1)
                    session_num += 1


class CMetricsTraceCallback(tf.keras.callbacks.Callback):
    def __init__(self, upd, best_cl, Cluster, feature, gnd, epochs):
        super(CMetricsTraceCallback, self).__init__()
        self.upd = upd
        self.best_db = best_cl
        self.Cluster = Cluster
        self.feature = feature
        self.gnd = gnd
        self.epochs = epochs

        self.best_acc = -1
        self.best_nmi = -1
        self.best_f1 = -1
        self.best_ari = -1
        self.best_epoch = -1
        self.cluster_centers = self.Cluster.cluster_centers_

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % self.upd == 0 or epoch == self.epochs - 1:
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


def kspace(args, feature, X, gnd):
    save_location = 'output/{}_{}_{}_power{}_epochs{}_dims{}-batch{}-lr{}-drop{}'.format(args.input, args.method,
        args.model, args.power, args.epochs, ",".join([str(x) for x in args.dims]), args.batch_size, args.learning_rate, args.dropout)

    # CREATE MODEL
    if args.model == 'ae':
        model = AE(dims=args.dims, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=Adam(lr=args.learning_rate), loss=MeanSquaredError())
    elif args.model == 'vae':
        model = VAE(dims=args.dims, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=Adam(lr=args.learning_rate))
    elif args.model == 'dae':
        model = DAE(dims=args.dims, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=Adam(lr=args.learning_rate), loss=MeanSquaredError())
    # else:
    #     model = DVAE(hidden1_dim=args.hidden, hidden2_dim=args.dimension, output_dim=X.shape[1], dropout=args.dropout)
    #     model.compile(optimizer=Adam(lr=args.learning_rate))

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


def run_kspace(args):
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
        kspace(args, feature, X, gnd)


def train(args, feature, X, gnd, model):
    """
    Model training
    :param args: cli arguments
    :param feature: original feature matrix
    :param X: the smoothed matrix
    :param gnd: ground truth labels
    :param model: the nn to be trained
    """

    m = len(np.unique(gnd)) # number of clusters according to ground truth
    Cluster = KMeans(n_clusters=m)

    # ADD NOISE IN CASE OF DENOISING MODELS
    if args.model == 'ae' or args.model == 'vae':
        input = feature
    else:
        input = salt_and_pepper(feature)

    # CALLBACKS
    es = EarlyStopping(monitor='loss', patience=args.early_stopping)

    # TRAINING
    # input is the plain feature matrix and output is the k-order convoluted. The model reconstructs the convolution!
    print('Training model for {}-order convolution'.format(args.power))
    model.fit(input, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es], verbose=1)

    embeds = model.embed(input)
    db, acc, nmi, f1, ari, centers = clustering(Cluster, embeds, gnd)
    print("Optimization Finished!")
    print("DB: {} ACC: {} F1: {} NMI: {} ARI: {}".format(db, acc, f1, nmi, ari))

    return model, centers, acc, nmi, f1, ari


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
    m = len(np.unique(gnd))  # number of clusters according to ground truth
    Cluster = KMeans(n_clusters=m)

    # ADD NOISE IN CASE OF DENOISING MODELS
    if args.model == 'ae' or args.model == 'vae':
        input = feature
    else:
        input = salt_and_pepper(feature)

    optimizer = Adam(lr=args.learning_rate)
    loss_fn = KLDivergence()

    train_dataset = tf.data.Dataset.from_tensor_slices((input, gnd))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

    centers = tf.Variable(initial_value=centers, trainable=True, dtype=tf.float32)
    # for each distribution create a list as train set is split into batches
    Q = list()
    P = list()
    epoch_loses = list()
    for i in range(args.c_epochs):
        epoch_loss = list()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                z = model.encoder(x_batch_train, training=True)
                z = tf.reshape(z, [tf.shape(z)[0], 1, tf.shape(z)[1]])  # reshape for broadcasting

                partial = tf.math.pow(tf.norm(z - centers, axis=2, ord='euclidean'), 2)
                nominator = 1 / (1 + partial)
                denominator = tf.math.reduce_sum(1 / (1 + partial))
                Q.insert(step, nominator / denominator)

                if i % 5 == 0:
                    partial = tf.math.pow(Q[step], 2) / tf.math.reduce_sum(Q[step], axis=1, keepdims=True)
                    nominator = partial
                    denominator = tf.math.reduce_sum(partial, axis=0)
                    P.insert(step, nominator / denominator)

                y_pred = model(x_batch_train, training=True)
                loss = model.compiled_loss(y_batch_train, y_pred)
                loss += loss_fn(P[step], Q[step])
                epoch_loss.append(loss)
            gradients = tape.gradient(loss, model.trainable_variables + [centers])
            optimizer.apply_gradients(zip(gradients, model.trainable_variables + [centers]))

        epoch_loss = np.mean(epoch_loss)
        epoch_loses.append(epoch_loss)
        print('Epoch: %d Loss: %.4f' % (i, float(epoch_loss)))

    embeds = model.embed(input)
    db, acc, nmi, f1, ari, _ = clustering(Cluster, embeds, gnd)
    print("DB: {} ACC: {} F1: {} NMI: {} ARI: {}".format(db, acc, f1, nmi, ari))

    return model, acc, nmi, f1, ari


def self_supervise_clusterBooster(args, feature, X, gnd, model, centers):
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
    model.compile(optimizer=Adam(lr=args.learning_rate), loss=KLDivergence())

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
