from sklearn.manifold import TSNE
from textwrap import wrap
from matplotlib import cm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

plt.rcParams.update({'font.size': 18, 'legend.fontsize': 16, 'lines.linewidth': 2})

colormap = cm.get_cmap('Spectral', 3)
colors = colormap(range(3))[::-1]


def tsne(embeds, gnd, args, epoch=-1):
    figure = plt.figure(figsize=(11.7, 8.27))
    palette = sns.color_palette("bright", len(np.unique(gnd)))
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(embeds)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=gnd, legend='full', palette=palette)
    if epoch == -1:
        plt.title('T-SNE')
    else:
        plt.title('T-SNE, epoch: {}'.format(epoch))
    plt.tight_layout()
    dims = '_'.join([str(v) for v in args.dims])
    directory = 'figures/kspace/tsne/{}/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims'.format(args.method, args.alpha, args.a_max, args.temperature, args.epochs, args.learning_rate, args.dropout, dims)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if epoch == -1:
        plt.savefig('{}/total.png'.format(directory), format='png')
    else:
        plt.savefig('{}/epoch_{}.png'.format(directory, epoch), format='png')


def plot_centers(embeds, centers, gnd, args, epoch=-1):
    figure = plt.figure(figsize=(11.7, 8.27))
    palette = sns.color_palette("bright", len(np.unique(gnd)))
    data = np.concatenate((embeds, centers), axis=0)
    tsne = TSNE(n_components=2, perplexity=30)
    data_embedded = tsne.fit_transform(data)

    X_embedded = data_embedded[:-centers.shape[0], :]
    centers_embedded = data_embedded[-centers.shape[0]:, :]

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=gnd, legend='full', palette=palette)
    plt.scatter(centers_embedded[:, 0], centers_embedded[:, 1], c='black', marker='X')
    if epoch == -1:
        plt.title('T-SNE')
    else:
        plt.title('T-SNE, epoch: {}'.format(epoch))
    plt.tight_layout()

    dims = '_'.join([str(v) for v in args.dims])
    if args.method == 'kspace_gnn':
        directory = 'figures/{}/tsne/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims'.format(args.method, args.alpha, args.a_max, args.temperature, args.epochs, args.learning_rate, args.dropout, dims)
    else:
        directory = 'figures/{}/tsne/{}_a_{}_b_{}_g_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims'.format(args.method, args.alpha, args.beta, args.gamma, args.temperature, args.epochs, args.learning_rate, args.dropout, dims)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if epoch == -1:
        plt.savefig('{}/total.png'.format(directory), format='png')
    else:
        plt.savefig('{}/epoch_{}.png'.format(directory, epoch), format='png')


def plot_results(config, pivot='Learning Rate'):

    filepath = 'figures/kspace/{}'.format(config['Model'])
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    data = pd.read_csv('output/kspace/results.csv')
    for k, v in config.items():
        data = data.loc[data[k] == v]

    data = data.sort_values(pivot)

    unique_xaxis = np.unique(data[pivot])

    if config['Dataset'] == 'cora':
        agc_acc = len(unique_xaxis) * [0.6892]
        agc_f1 = len(unique_xaxis) * [0.6561]
        agc_nmi = len(unique_xaxis) * [0.5368]
    elif config['Dataset'] == 'citeseer':
        agc_acc = len(unique_xaxis) * [0.6700]
        agc_f1 = len(unique_xaxis) * [0.6248]
        agc_nmi = len(unique_xaxis) * [0.4113]
    elif config['Dataset'] == 'pubmed':
        agc_acc = len(unique_xaxis) * [0.6978]
        agc_f1 = len(unique_xaxis) * [0.6872]
        agc_nmi = len(unique_xaxis) * [0.3159]
    else:
        return

    acc_means = data.groupby(pivot, as_index=False)['Accuracy'].mean()
    nmi_means = data.groupby(pivot, as_index=False)['NMI'].mean()
    f1s_means = data.groupby(pivot, as_index=False)['F1'].mean()

    acc_stds = data.groupby(pivot, as_index=False)['Accuracy'].std()
    nmi_stds = data.groupby(pivot, as_index=False)['NMI'].std()
    f1_stds = data.groupby(pivot, as_index=False)['F1'].std()

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(acc_means[pivot], acc_means['Accuracy'], color='purple', label='Acc', marker='x')
    ax[0].fill_between(acc_means[pivot], acc_means['Accuracy'] - acc_stds['Accuracy'], acc_means['Accuracy'] + acc_stds['Accuracy'], color='purple', alpha=0.2)
    ax[0].plot(acc_means[pivot], agc_acc, color='black', label='AGC')
    ax[0].legend()

    ax[1].plot(f1s_means[pivot], f1s_means['F1'], color='yellow', label='F1', marker='x')
    ax[1].fill_between(f1s_means[pivot], f1s_means['F1'] - f1_stds['F1'], f1s_means['F1'] + f1_stds['F1'], color='yellow', alpha=0.2)
    ax[1].plot(f1s_means[pivot], agc_f1, color='black', label='AGC')
    ax[1].legend()

    ax[2].plot(nmi_means[pivot], nmi_means['NMI'], color='green', label='NMI', marker='x')
    ax[2].fill_between(nmi_means[pivot], nmi_means['NMI'] - nmi_stds['NMI'], nmi_means['NMI'] + nmi_stds['NMI'], color='green', alpha=0.2)
    ax[2].plot(nmi_means[pivot], agc_nmi, color='black', label='AGC')
    ax[2].legend()

    ax[0].set_xlabel(pivot)
    ax[1].set_xlabel(pivot)
    ax[2].set_xlabel(pivot)
    ax[0].set_ylabel('Score')
    ax[1].set_ylabel('Score')
    ax[2].set_ylabel('Score')

    num_items = len(config.keys())
    title = ''
    for i, (k, v) in enumerate(config.items()):
        if i == num_items - 1:
            title += k + ':' + str(v)
        else:
            title += k + ':' + str(v) + ', '
    plt.suptitle("\n".join(wrap(title, 75)))

    filepath = filepath + '/'
    for i, v in enumerate(config.values()):
        if i == num_items - 1:
            filepath += str(v)
        else:
            filepath += str(v) + '_'
    filepath += '.png'
    plt.savefig(filepath, format='png')
    plt.show()


def plot_temperature_losses(temps, alpha, a_max, epochs, lr, dropout, dims):
    dirs = []
    dims = '_'.join([str(v) for v in dims])
    for temp in temps:
        dirs.append('../pickles/kspace_gnn/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims'.format(alpha, a_max, temp, epochs, lr, dropout, dims))

    closs_files = {}
    recloss_files = {}  # key = temperature value = list of files
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk(dir))
        closs_files[temps[idx]] = [file for file in files if 'clustlosses' in file]
        recloss_files[temps[idx]] = [file for file in files if 'reclosses' in file]

    for temp in closs_files.keys():  # each iter different temp
        c_losses = []
        rec_losses = []
        for i in range(len(closs_files[temp])):
            clust = pickle.load(open('../pickles/kspace_gnn/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims/clustlosses_{}.pickle'.format(alpha, a_max, temp, epochs, lr, dropout, dims, i), 'rb'))
            rec = pickle.load(open('../pickles/kspace_gnn/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims/reclosses_{}.pickle'.format(alpha, a_max, temp, epochs, lr, dropout, dims, i), 'rb'))
            # convert tensors to float
            clust = [float(i) for i in clust]
            rec = [float(i) for i in rec]
            # append to iteration lists
            c_losses.append(clust)
            rec_losses.append(rec)

        # convert to numpy for ease
        c_losses = np.array(c_losses)
        rec_losses = np.array(rec_losses)

        # calc stats
        mean_c_losses = np.mean(c_losses, axis=0)
        mean_rec_losses = np.mean(rec_losses, axis=0)
        std_c_losses = np.std(c_losses, axis=0)
        std_rec_losses = np.std(rec_losses, axis=0)

        # create the utmost dictionary for each freq
        closs_files[temp] = {'mean': mean_c_losses, 'std': std_c_losses}
        recloss_files[temp] = {'mean': mean_rec_losses, 'std': std_rec_losses}

    directory = 'figures/kspace_gnn/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims/losses'.format(alpha, a_max, temp, epochs, lr, dropout, dims)
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, temp in enumerate(list(closs_files.keys())[::-1]):
        ax.plot(closs_files[temp]['mean'], label='temp={}'.format(temp), color=colors[idx])
        ax.fill_between(np.arange(epochs), closs_files[temp]['mean'] - closs_files[temp]['std'], closs_files[temp]['mean'] + closs_files[temp]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cluster loss')
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_ylim(-0.01, 0.175)
    ax.grid(axis='y')
    plt.legend(loc='upper right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/clust_loss.pdf'.format(directory), format='pdf')
    plt.show()
    print(recloss_files)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, temp in enumerate(list(recloss_files.keys())[::-1]):
        ax.plot(recloss_files[temp]['mean'], label='temp={}'.format(temp), color=colors[idx])
        ax.fill_between(np.arange(epochs), recloss_files[temp]['mean'] - recloss_files[temp]['std'], recloss_files[temp]['mean'] + recloss_files[temp]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rec loss')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('{}/rec_loss.pdf'.format(directory), format='pdf')
    plt.show()
# plot_temperature_losses([30, 50, 100], 'linear', 0.9, 2000, 0.001, 0.2, [32])


def plot_temperature_metrics(temps, alpha, a_max, epochs, lr, dropout, dims):
    dirs = []
    dims = '_'.join([str(v) for v in dims])
    for temp in temps:
        dirs.append('../pickles/kspace_gnn/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims'.format(alpha, a_max, temp, epochs, lr, dropout, dims))

    acc_files = {}
    ari_files = {}  # key = temperature value = list of files
    nmi_files = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk(dir))
        acc_files[temps[idx]] = [file for file in files if 'accs' in file]
        ari_files[temps[idx]] = [file for file in files if 'aris' in file]
        nmi_files[temps[idx]] = [file for file in files if 'nmis' in file]

    for temp in acc_files.keys():  # each iter different temp
        accs = []
        aris = []
        nmis = []
        for i in range(len(acc_files[temp])):
            acc = pickle.load(open('../pickles/kspace_gnn/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims/accs_{}.pickle'.format(alpha, a_max, temp, epochs, lr, dropout, dims, i), 'rb'))
            ari = pickle.load(open('../pickles/kspace_gnn/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims/aris_{}.pickle'.format(alpha, a_max, temp, epochs, lr, dropout, dims, i), 'rb'))
            nmi = pickle.load(open('../pickles/kspace_gnn/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims/nmis_{}.pickle'.format(alpha, a_max, temp, epochs, lr, dropout, dims, i), 'rb'))
            # convert tensors to float
            acc = [float(i) for i in acc]
            ari = [float(i) for i in ari]
            nmi = [float(i) for i in nmi]
            # append to iteration lists
            accs.append(acc)
            aris.append(ari)
            nmis.append(nmi)

        # convert to numpy for ease
        accs = np.array(accs)
        aris = np.array(aris)
        nmis = np.array(nmis)

        # calc stats
        mean_acc = np.mean(accs, axis=0)
        mean_ari = np.mean(aris, axis=0)
        mean_nmi = np.mean(nmis, axis=0)
        std_acc = np.std(accs, axis=0)
        std_ari = np.std(aris, axis=0)
        std_nmi = np.std(nmis, axis=0)

        # create the utmost dictionary for each freq
        acc_files[temp] = {'mean': mean_acc, 'std': std_acc}
        ari_files[temp] = {'mean': mean_ari, 'std': std_ari}
        nmi_files[temp] = {'mean': mean_nmi, 'std': std_nmi}

    directory = '../figures/kspace_gnn/performance/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims/losses'.format(alpha, a_max, temp, epochs, lr, dropout, dims)
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, temp in enumerate(list(acc_files.keys())[::-1]):
        ax.plot(acc_files[temp]['mean'], label='temp={}'.format(temp), color=colors[idx])
        ax.fill_between(np.arange(epochs), acc_files[temp]['mean'] - acc_files[temp]['std'], acc_files[temp]['mean'] + acc_files[temp]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_yticks([0.00, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.01, 1.01)
    ax.grid(axis='y')
    plt.legend(loc='upper right', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/acc.pdf'.format(directory), format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, temp in enumerate(list(ari_files.keys())[::-1]):
        ax.plot(ari_files[temp]['mean'], label='temp={}'.format(temp), color=colors[idx])
        ax.fill_between(np.arange(epochs), ari_files[temp]['mean'] - ari_files[temp]['std'], ari_files[temp]['mean'] + ari_files[temp]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ARI')
    ax.set_yticks([0.00, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.01, 1.01)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('{}/ari.pdf'.format(directory), format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, temp in enumerate(list(nmi_files.keys())[::-1]):
        ax.plot(nmi_files[temp]['mean'], label='temp={}'.format(temp), color=colors[idx])
        ax.fill_between(np.arange(epochs), nmi_files[temp]['mean'] - nmi_files[temp]['std'], nmi_files[temp]['mean'] + nmi_files[temp]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NMI')
    ax.set_yticks([0.00, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.01, 1.01)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('{}/nmi.pdf'.format(directory), format='pdf')
    plt.show()
# plot_temperature_metrics([30, 50, 100], 'linear', 0.5, 2000, 0.001, 0.2, [32])
