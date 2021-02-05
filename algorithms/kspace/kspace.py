import csv
import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.metrics import clustering
from utils.utils import load_data_trunc
import tensorflow as tf
from .models import AE, VAE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, KLDivergence
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from utils.utils import save_results, salt_and_pepper, largest_eigval_smoothing_filter, preprocess_adj
from .utils import AlphaRateScheduler, alpha_scheduler, get_alpha


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


def kspace(args, feature, X, gnd):
    save_location = 'output/{}_{}_{}_power{}_epochs{}_dims{}-batch{}-lr{}-drop{}'.format(args.input, args.method,
        args.model, args.power, args.epochs, ",".join([str(x) for x in args.dims]), args.batch_size, args.learning_rate, args.dropout)

    m = len(np.unique(gnd))
    alphas = get_alpha(args.a_max, args.epochs, args.alpha)

    # CREATE MODEL
    if args.model == 'ae' or args.model == 'dae':
        model = AE(dims=args.dims, output_dim=X.shape[1], dropout=args.dropout, num_centers=m, alphas=alphas)
        model.compile(optimizer=Adam(lr=args.learning_rate), loss=MeanSquaredError())
    else:  # args.model == 'vae' or args.model == 'dvae'
        model = VAE(dims=args.dims, output_dim=X.shape[1], dropout=args.dropout)
        model.compile(optimizer=Adam(lr=args.learning_rate))

    # TRAINING OR LOAD MODEL IF IT EXISTS
    if not os.path.exists(save_location):
        model, centers, acc, nmi, f1, ari = train(args, feature, X, gnd, model)

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
    csv_location = 'output/kspace/logs/{}amax_{}step.csv'.format(args.a_max, args.alpha)
    csv_logger = CSVLogger(csv_location)
    alpha_cb = AlphaRateScheduler(alpha_scheduler)

    # TRAINING
    # input is the plain feature matrix and output is the k-order convoluted. The model reconstructs the convolution!
    print('Training model for {}-order convolution'.format(args.power))
    model.fit(input, X, epochs=args.epochs, batch_size=args.batch_size, shuffle=True, callbacks=[es, csv_logger, alpha_cb], verbose=1)

    embeds = model.embed(input)
    db, acc, nmi, f1, ari, centers = clustering(Cluster, embeds, gnd)
    print("Optimization Finished!")
    print("DB: {} ACC: {} F1: {} NMI: {} ARI: {}".format(db, acc, f1, nmi, ari))

    return model, centers, acc, nmi, f1, ari


def write_results(args, ac, nm, f1, ari):
    file_exists = os.path.isfile('output/kspace/results.csv')
    with open('output/kspace/results.csv', 'a') as f:
        columns = ['Dataset', 'Model', 'Dimensions', 'Epochs', 'Batch Size', 'Learning Rate', 'Dropout',
                   'A Max', 'A Type', 'Power', 'Accuracy', 'NMI', 'F1', 'ARI']
        writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=columns)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow({'Dataset': args.input, 'Model': args.model, 'Dimensions': ",".join([str(x) for x in args.dims]),
                         'Epochs': args.epochs, 'Batch Size': args.batch_size, 'Learning Rate': args.learning_rate,
                         'Dropout': args.dropout, 'A Max': args.a_max, 'A Type': args.alpha, 'Power': args.power,
                         'Accuracy': ac, 'NMI': nm, 'F1': f1, 'ARI': ari})