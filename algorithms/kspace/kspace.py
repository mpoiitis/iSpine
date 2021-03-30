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
from .utils import cluster_kl_loss as cluster_loss
# @pavlos # 
from .utils import cluster_q_loss as cluster_q_loss


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

    train_dataset = tf.data.Dataset.from_tensor_slices((feature, X))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

    mse_loss = tf.keras.losses.MeanSquaredError()

    tb_loss = list()
    tb_c_loss = list()
    tb_rec_loss = list()
    p = list()
    q = list()

    # @pavlos 
    # flag = 1
    for epoch in range(args.epochs):
        epoch_loss = list()
        rec_epoch_loss = list()
        clust_epoch_loss = list()

        if epoch == (args.slack - 1):
            z = model.encoder(feature)
            kmeans = KMeans(n_clusters=m)
            predicted = kmeans.fit_predict(z)

            acc, nmi, ari, f1 = calc_metrics(predicted, gnd)
            print('Acc={:.4f}% Nmi={:.4f}% Ari={:.4f}% Macro-f1={:.4f}%'.format(acc * 100, nmi * 100, ari * 100,
                                                                                f1 * 100))
            
            model.centers.assign(kmeans.cluster_centers_)
            plot_centers(z, kmeans.cluster_centers_, gnd, epoch)


        if epoch == 0 or epoch % 50 == 0 or epoch == (args.epochs - 1):
            z = model.embed(feature)
            centers = model.centers.numpy()
            plot_centers(z, centers, gnd, epoch)
            m_predicted = model.predict(feature).numpy()
            acc, nmi, ari, f1 = calc_metrics(m_predicted, gnd)
            print('\t\t\t\tAcc={:.4f}% Nmi={:.4f}% Ari={:.4f}% Macro-f1={:.4f}%'.format(acc * 100, nmi * 100, ari * 100,
                                                                        f1 * 100))

        
        # @pavlos 
        # lll = ['weights','centers']
        # if  not (epoch % 25):
        #     flag = 1-flag
        #     print('only {}'.format(lll[flag]))

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                z = model.encoder(x_batch_train)
                y_pred = model.decoder(z)
                rec_loss = mse_loss(y_batch_train, y_pred)

                if epoch >= args.slack - 1:
                    
                    # @pavlos 
                    # c_loss = cluster_loss(z, model.centers)
                    # loss = rec_loss + args.a_max * c_loss # alphas[epoch] * c_loss
                    c_loss = tf.reduce_mean(cluster_q_loss(z, model.centers))
                    if epoch < args.slack - 1:
                        correction_factor = 1
                    if epoch == args.slack - 1:
                        correction_factor = abs(rec_loss / c_loss)
                    c_loss = c_loss*correction_factor
                    loss = rec_loss + alphas[epoch] * c_loss
                else:
                    loss = rec_loss

            if epoch >= args.slack - 1:
                # @pavlos 
                # gradients = tape.gradient(loss, model.trainable_weights)
                # model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                if epoch %2 :
                    gradients = tape.gradient(loss, [model.centers])
                    gradients =  [0.00001*i for i in gradients]
                    model.optimizer.apply_gradients(zip(gradients, [model.centers]))
                else:
                    gradients = tape.gradient(loss, model.encoder.trainable_weights + model.decoder.trainable_weights)
                    model.optimizer.apply_gradients(zip(gradients, model.encoder.trainable_weights + model.decoder.trainable_weights))

                epoch_loss.append(loss)
                rec_epoch_loss.append(rec_loss)
                clust_epoch_loss.append(c_loss)

            else:
                gradients = tape.gradient(loss, model.encoder.trainable_weights + model.decoder.trainable_weights)
                model.optimizer.apply_gradients(zip(gradients, model.encoder.trainable_weights + model.decoder.trainable_weights))

                epoch_loss.append(loss)
                rec_epoch_loss.append(rec_loss)

        if epoch >= args.slack - 1:
            e_loss = np.mean(epoch_loss)
            r_e_loss = np.mean(rec_epoch_loss)
            c_e_loss = np.mean(clust_epoch_loss)
            tb_loss.append(e_loss)
            tb_c_loss.append(c_e_loss)
            tb_rec_loss.append(r_e_loss)
            print('Epoch: {}    Loss: {:.4f} Reconstruction: {:.4f} Clustering: {:.4f}'.format(epoch, 100*e_loss, 100*r_e_loss, 100*c_e_loss))
        else:
            e_loss = np.mean(epoch_loss)
            r_e_loss = np.mean(rec_epoch_loss)
            tb_loss.append(e_loss)
            tb_rec_loss.append(r_e_loss)
            tb_c_loss.append(np.nan)
            print('Epoch: {}    Loss: {:.4f} Reconstruction: {:.4f}'.format(epoch, 100*e_loss, 100*r_e_loss))


    # @pavlos #
    z = model.encoder(feature)
    kmeans = KMeans(n_clusters=m)
    predicted = kmeans.fit_predict(z)
    acc, nmi, f1, ari = calc_metrics(predicted, gnd)

    # predicted = model.predict(feature)
    # acc, nmi, f1, ari = calc_metrics(predicted.numpy(), gnd)
    print("Optimization Finished!")
    print('Acc= {:.4f}%    Nmi= {:.4f}%    Ari= {:.4f}%   Macro-f1= {:.4f}%'.format(acc * 100, nmi * 100, ari * 100, f1 * 100))
    # @pavlos #
    m_predicted = model.predict(feature).numpy()
    acc, nmi, ari, f1 = calc_metrics(m_predicted, gnd)
    print('Acc={:.4f}% Nmi={:.4f}% Ari={:.4f}% Macro-f1={:.4f}%'.format(acc * 100, nmi * 100, ari * 100,
                                                                        f1 * 100))

    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(11.7, 8.27))
    plt.plot(tb_loss, color='black', label='total')
    plt.plot(tb_rec_loss, color='red', label='reconstruction')
    plt.plot(tb_c_loss, color='blue', label='clustering')
    plt.title('Losses for each epoch')
    plt.xlabel('Epoch')
    # plt.ylabel('Log Value')
    # plt.yscale('log')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('figures/kspace/tsne/epochs/losses.png', format='png')

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
