import csv
import os
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.cluster import KMeans
from .models import AE, VAE
from tensorflow.keras.optimizers import Adam
from utils.utils import load_data_trunc, save_results, largest_eigval_smoothing_filter, preprocess_adj
from .utils import get_alpha, calc_metrics
from utils.plots import plot_centers
from .utils import ClusterLoss

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


def run_kspace(args):
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


def kspace(args, feature, X, gnd):
    save_location = 'output/{}_{}_power{}_epochs{}_dims{}-batch{}-lr{}-drop{}'.format(args.input, args.method, args.power,
                                                                                         args.epochs, ",".join(
            [str(x) for x in args.dims]), args.batch_size, args.learning_rate, args.dropout)

    m = len(np.unique(gnd))
    # CREATE MODEL
    model = AE(dims=args.dims, output_dim=X.shape[1], dropout=args.dropout, num_centers=m)
    model.compile(optimizer=Adam(lr=args.learning_rate))

    # TRAINING OR LOAD MODEL IF IT EXISTS
    if not os.path.exists(save_location):
        model, acc, nmi, f1, ari = train(args, feature, X, gnd, model)

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


def train(args, feature, X, gnd, model):
    """
    Model training
    :param args: cli arguments
    :param feature: original feature matrix
    :param X: the smoothed matrix
    :param gnd: ground truth labels
    :param model: the nn to be trained
    """
    alphas = get_alpha(args.a_max, args.epochs, args.slack, args.alpha)

    m = len(np.unique(gnd))  # number of clusters according to ground truth

    # ADD NOISE IN CASE OF DENOISING MODELS
    feature

    train_dataset = tf.data.Dataset.from_tensor_slices((feature, X))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

    mse_loss = tf.keras.losses.MeanSquaredError()
    cluster_loss = ClusterLoss()

    for epoch in range(args.epochs):
        epoch_loss = list()
        rec_epoch_loss = list()
        clust_epoch_loss = list()

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                z = model.encoder(x_batch_train)
                y_pred = model.decoder(z)
                rec_loss = mse_loss(y_batch_train, y_pred)
                c_loss = cluster_loss(z, model.centers)
                loss = rec_loss + alphas[epoch] * c_loss

            gradients = tape.gradient(loss, model.trainable_weights)
            # Update weights
            model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            epoch_loss.append(loss)
            rec_epoch_loss.append(rec_loss)
            clust_epoch_loss.append(c_loss)

        if epoch == (args.slack - 1):
            z = model.encoder(feature)
            kmeans = KMeans(n_clusters=m)
            predicted = kmeans.fit_predict(z)

            acc, nmi, ari, f1 = calc_metrics(predicted, gnd)
            print('Acc={:.4f}% Nmi={:.4f}% Ari={:.4f}% Macro-f1={:.4f}%'.format(acc * 100, nmi * 100, ari * 100, f1 * 100))
            model.centers.assign(kmeans.cluster_centers_)

        print('Epoch: {}    Loss: {:.4f} Reconstruction: {:.4f} Clustering: {:.4f}'.format(epoch, np.mean(epoch_loss), np.mean(rec_epoch_loss), np.mean(clust_epoch_loss)))

        if epoch % (args.slack - 1) == 0 or epoch == (args.epochs - 1):
            z = model.embed(feature)
            centers = model.centers.numpy()
            plot_centers(z, centers, gnd, epoch)

    # z = model.encoder(input)
    # kmeans = KMeans(n_clusters=m)
    # predicted = kmeans.fit_predict(z)
    # acc, nmi, f1, ari = calc_metrics(predicted, gnd)

    predicted = model.predict(input)
    acc, nmi, f1, ari = calc_metrics(predicted.numpy(), gnd)
    print("Optimization Finished!")
    print('Acc= {:.4f}%    Nmi= {:.4f}%    Ari= {:.4f}%   Macro-f1= {:.4f}%'.format(acc * 100, nmi * 100, ari * 100, f1 * 100))

    return model, acc, nmi, f1, ari


def write_results(args, ac, nm, f1, ari):
    file_exists = os.path.isfile('output/kspace/results.csv')
    with open('output/kspace/results.csv', 'a') as f:
        columns = ['Dataset', 'Dimensions', 'Epochs', 'Batch Size', 'Learning Rate', 'Dropout',
                   'A Max', 'A Type', 'Power', 'Accuracy', 'NMI', 'F1', 'ARI']
        writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=columns)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(
            {'Dataset': args.input, 'Dimensions': ",".join([str(x) for x in args.dims]),
             'Epochs': args.epochs, 'Batch Size': args.batch_size, 'Learning Rate': args.learning_rate,
             'Dropout': args.dropout, 'A Max': args.a_max, 'A Type': args.alpha, 'Power': args.power,
             'Accuracy': ac, 'NMI': nm, 'F1': f1, 'ARI': ari})
