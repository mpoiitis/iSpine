from sklearn.manifold import TSNE
from textwrap import wrap
from tqdm import tqdm
from matplotlib import cm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

plt.rcParams.update({'font.size': 18, 'legend.fontsize': 16, 'lines.linewidth': 2})

# colormap = cm.get_cmap('RdYlGn_r', 3)
# colors = colormap(range(3))[::-1]
colors = ['#5e3c99', '#b2abd2', '#fdb863', '#e66101']


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
    data = np.concatenate((embeds, centers), axis=0)
    tsne = TSNE(n_components=2, perplexity=30)
    data_embedded = tsne.fit_transform(data)

    X_embedded = data_embedded[:-centers.shape[0], :]
    centers_embedded = data_embedded[-centers.shape[0]:, :]

    flatui = ["#f43f1a", "#00c0ff", "#ffc100", "#ad6aea", "#68d16a", "#f3b5f4", "#8dd3c7"]
    sns.set_palette(flatui)
    palette = sns.color_palette()
    palette = palette[:len(np.unique(gnd))]
    # palette = sns.color_palette('Set2', len(np.unique(gnd)))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=gnd, legend=None, palette=palette)
    ax.scatter(centers_embedded[:, 0], centers_embedded[:, 1], c='black', marker='X', s=250)

    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.axis('equal')
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.tight_layout()

    dims = '_'.join([str(v) for v in args.dims])
    if args.method == 'kspace_gnn':
        directory = 'figures/{}/tsne/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims'.format(args.method, args.alpha, args.a_max, args.temperature, args.epochs, args.learning_rate, args.dropout, dims)
    else:
        directory = 'figures/{}/tsne/{}_{}_{}_a_{}_a_fun_{}_b _{}_b_fun_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims_{}_power'.format(args.method, args.enc, args.input, args.alpha, args.a_prog, args.beta, args.b_prog, args.temperature, args.epochs, args.learning_rate, args.dropout, dims, args.power)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if epoch == -1:
        plt.savefig('{}/total.png'.format(directory), format='png', bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}/epoch_{}.png'.format(directory, epoch), format='png', bbox_inches='tight', pad_inches=0)


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


def plot_ks(dataset):
    point_df = pd.read_csv('../output/k/pointNet_{}_k.csv'.format(dataset))
    mlp_df = pd.read_csv('../output/k/MLP_{}_k.csv'.format(dataset))
    cnn_df = pd.read_csv('../output/k/CNN_{}_k.csv'.format(dataset))

    acc_means = dict()
    ari_means = dict()
    nmi_means = dict()
    f1_means = dict()
    acc_stds = dict()
    ari_stds = dict()
    nmi_stds = dict()
    f1_stds = dict()

    means = point_df.groupby('Power').mean().reset_index()
    stds = point_df.groupby('Power').std().reset_index()

    acc_means['pointNet'] = means.sort_values('Power')['Acc']
    ari_means['pointNet'] = means.sort_values('Power')['Ari']
    nmi_means['pointNet'] = means.sort_values('Power')['Nmi']
    f1_means['pointNet'] = means.sort_values('Power')['F1']
    acc_stds['pointNet'] = stds.sort_values('Power')['Acc']
    ari_stds['pointNet'] = stds.sort_values('Power')['Ari']
    nmi_stds['pointNet'] = stds.sort_values('Power')['Nmi']
    f1_stds['pointNet'] = stds.sort_values('Power')['F1']

    means = mlp_df.groupby('Power').mean().reset_index()
    stds = mlp_df.groupby('Power').std().reset_index()

    acc_means['mlp'] = means.sort_values('Power')['Acc']
    ari_means['mlp'] = means.sort_values('Power')['Ari']
    nmi_means['mlp'] = means.sort_values('Power')['Nmi']
    f1_means['mlp'] = means.sort_values('Power')['F1']
    acc_stds['mlp'] = stds.sort_values('Power')['Acc']
    ari_stds['mlp'] = stds.sort_values('Power')['Ari']
    nmi_stds['mlp'] = stds.sort_values('Power')['Nmi']
    f1_stds['mlp'] = stds.sort_values('Power')['F1']

    means = cnn_df.groupby('Power').mean().reset_index()
    stds = cnn_df.groupby('Power').std().reset_index()

    acc_means['cnn'] = means.sort_values('Power')['Acc']
    ari_means['cnn'] = means.sort_values('Power')['Ari']
    nmi_means['cnn'] = means.sort_values('Power')['Nmi']
    f1_means['cnn'] = means.sort_values('Power')['F1']
    acc_stds['cnn'] = stds.sort_values('Power')['Acc']
    ari_stds['cnn'] = stds.sort_values('Power')['Ari']
    nmi_stds['cnn'] = stds.sort_values('Power')['Nmi']
    f1_stds['cnn'] = stds.sort_values('Power')['F1']

    directory = '../figures/pointSpectrum/k/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, key in enumerate(list(acc_means.keys())):
        ax.plot(np.arange(1, len(acc_means[key]) + 1), acc_means[key], label=key, color=colors[idx])
        ax.fill_between(np.arange(1, len(acc_means[key]) + 1), acc_means[key] - acc_stds[key], acc_means[key] + acc_stds[key], color=colors[idx], alpha=0.3)
    ax.set_xlabel('k')
    ax.set_ylabel('Accuracy')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(-0.01, 0.8)
    ax.grid(axis='y')
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/{}_acc.pdf'.format(directory, dataset), format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, key in enumerate(list(ari_means.keys())):
        ax.plot(np.arange(1, len(ari_means[key]) + 1), ari_means[key], label=key, color=colors[idx])
        ax.fill_between(np.arange(1, len(ari_means[key]) + 1), ari_means[key] - ari_stds[key], ari_means[key] + ari_stds[key], color=colors[idx], alpha=0.3)
    ax.set_xlabel('k')
    ax.set_ylabel('ARI')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(-0.01, 0.8)
    ax.grid(axis='y')
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/{}_ari.pdf'.format(directory, dataset), format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, key in enumerate(list(nmi_means.keys())):
        ax.plot(np.arange(1, len(nmi_means[key]) + 1), nmi_means[key], label=key, color=colors[idx])
        ax.fill_between(np.arange(1, len(nmi_means[key]) + 1), nmi_means[key] - nmi_stds[key], nmi_means[key] + nmi_stds[key], color=colors[idx], alpha=0.3)
    ax.set_xlabel('k')
    ax.set_ylabel('NMI')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(-0.01, 0.8)
    ax.grid(axis='y')
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/{}_nmi.pdf'.format(directory, dataset), format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, key in enumerate(list(f1_means.keys())):
        ax.plot(np.arange(1, len(f1_means[key]) + 1), f1_means[key], label=key, color=colors[idx])
        ax.fill_between(np.arange(1, len(f1_means[key]) + 1), f1_means[key] - f1_stds[key], f1_means[key] + f1_stds[key], color=colors[idx], alpha=0.3)
    ax.set_xlabel('k')
    ax.set_ylabel('F1')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(-0.01, 0.8)
    ax.grid(axis='y')
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/{}_f1.pdf'.format(directory, dataset), format='pdf')
    plt.show()
# plot_ks('cora')


def smooth(scalars, weight=0.95):
    """
    Smoothing of a list of values, similar to Tensorboard's smoothing
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_convergence(models, dataset):
    dirs = []
    for model in models:
        if dataset == 'cora':
            dirs.append('pickles/pointSpectrum/{}_{}_1.0_a_const_a_fun_1.0_b _const_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_8_power'.format(model, dataset))
        elif dataset == 'citeseer':
            dirs.append('pickles/pointSpectrum/{}_{}_1.0_a_expdec_a_fun_5.0_b _exp_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_6_power'.format(model, dataset))
        elif dataset == 'pubmed':
            dirs.append(
                'pickles/pointSpectrum/{}_{}_1.0_a_expdec_a_fun_1.0_b _exp_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_8_power'.format(model, dataset))
    loss_files = {}  # key = model value = list of files
    r_loss_files = {}
    c_loss_files = {}
    for idx, dir in enumerate(dirs):
        path, dirs, files = next(os.walk("../{}".format(dir)))
        loss_files[models[idx]] = [file for file in files if file.startswith('losses')]
        r_loss_files[models[idx]] = [file for file in files if 'reclosses' in file]
        c_loss_files[models[idx]] = [file for file in files if 'clustlosses' in file]

    for model in loss_files.keys():  # each iter different model
        losses = []
        r_losses = []
        c_losses = []
        for i in tqdm(range(len(loss_files[model]))):
            if dataset == 'cora':
                loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_const_a_fun_1.0_b _const_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_8_power/losses_{}.pickle'.format(model, dataset, i), 'rb'))
                r_loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_const_a_fun_1.0_b _const_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_8_power/reclosses_{}.pickle'.format(model, dataset, i), 'rb'))
                c_loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_const_a_fun_1.0_b _const_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_8_power/clustlosses_{}.pickle'.format(model, dataset, i), 'rb'))
            elif dataset == 'citeseer':
                loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_expdec_a_fun_5.0_b _exp_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_6_power/losses_{}.pickle'.format(model, dataset, i), 'rb'))
                r_loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_expdec_a_fun_5.0_b _exp_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_6_power/reclosses_{}.pickle'.format(model, dataset, i), 'rb'))
                c_loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_expdec_a_fun_5.0_b _exp_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_6_power/clustlosses_{}.pickle'.format(model, dataset, i), 'rb'))
            elif dataset == 'pubmed':
                loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_expdec_a_fun_1.0_b _exp_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_8_power/losses_{}.pickle'.format(model, dataset, i), 'rb'))
                r_loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_expdec_a_fun_1.0_b _exp_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_8_power/reclosses_{}.pickle'.format(model, dataset, i), 'rb'))
                c_loss = pickle.load(open('../pickles/pointSpectrum/{}_{}_1.0_a_expdec_a_fun_1.0_b _exp_b_fun_10_temperature_500_epochs_0.01_lr_0.2_dropout_100_dims_8_power/clustlosses_{}.pickle'.format(model, dataset, i), 'rb'))

            # convert tensors to float
            loss = [float(i) for i in loss]
            r_loss = [float(i) for i in r_loss]
            c_loss = [float(i) for i in c_loss]

            # append to iteration lists
            losses.append(loss)
            r_losses.append(r_loss)
            c_losses.append(c_loss)

        # convert to numpy for ease
        losses = np.array(losses)
        r_losses = np.array(r_losses)
        c_losses = np.array(c_losses)

        # calc stats
        mean_losses = np.mean(losses, axis=0)
        mean_r_losses = np.mean(r_losses, axis=0)
        mean_c_losses = np.mean(c_losses, axis=0)
        std_losses = np.std(losses, axis=0)
        std_r_losses = np.std(r_losses, axis=0)
        std_c_losses = np.std(c_losses, axis=0)

        # create the utmost dictionary for each freq
        loss_files[model] = {'mean': mean_losses, 'std': std_losses}
        r_loss_files[model] = {'mean': mean_r_losses, 'std': std_r_losses}
        c_loss_files[model] = {'mean': mean_c_losses, 'std': std_c_losses}

    directory = '../figures/pointSpectrum/convergence/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, model in enumerate(list(loss_files.keys())):
        smoothed = smooth(loss_files[model]['mean'])
        ax.plot(smoothed, label=model, color=colors[idx])
        # ax.fill_between(np.arange(1, len(loss_files[model]['mean']) + 1), loss_files[model]['mean'] - loss_files[model]['std'], loss_files[model]['mean'] + loss_files[model]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # ax.set_yticks([0.00, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    # ax.set_ylim(-0.01, 1.21)
    ax.grid(axis='y')
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/{}_loss.pdf'.format(directory, dataset), format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, model in enumerate(list(r_loss_files.keys())):
        smoothed = smooth(r_loss_files[model]['mean'])
        ax.plot(smoothed, label=model, color=colors[idx])
        # ax.fill_between(np.arange(1, len(r_loss_files[model]['mean']) + 1), r_loss_files[model]['mean'] - r_loss_files[model]['std'], r_loss_files[model]['mean'] + r_loss_files[model]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction loss')
    # ax.set_yticks([0.00, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    # ax.set_ylim(-0.01, 1.21)
    ax.grid(axis='y')
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/{}_r_loss.pdf'.format(directory, dataset), format='pdf')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
    for idx, model in enumerate(list(c_loss_files.keys())):
        smoothed = smooth(c_loss_files[model]['mean'])
        ax.plot(smoothed, label=model, color=colors[idx])
        # ax.fill_between(np.arange(1, len(c_loss_files[model]['mean']) + 1), c_loss_files[model]['mean'] - c_loss_files[model]['std'], c_loss_files[model]['mean'] + c_loss_files[model]['std'], color=colors[idx], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Clustering loss')
    # ax.set_yticks([0.00, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    # ax.set_ylim(-0.01, 1.21)
    ax.grid(axis='y')
    plt.legend(loc='best', ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig('{}/{}_c_loss.pdf'.format(directory, dataset), format='pdf')
    plt.show()
# plot_convergence(['pointNet', 'mlp', 'cnn'], 'cora')


def plot_features():

    directory = '../output/feature_importance'
    path, dirs, files = next(os.walk(directory))

    models = list()
    classes = list()
    accs = list()
    nmis = list()
    aris = list()
    f1s = list()
    for file in files:
        df = pd.read_csv(directory + '/' + file)
        parts = file.split('_')
        models.append(parts[0])
        if 'full' in parts[-1]:
            classes.append(parts[-1][:-4])
        else:
            classes.append('-' + parts[-1][:-4])
        accs.append(df['Acc'].mean())
        nmis.append(df['Nmi'].mean())
        aris.append(df['Ari'].mean())
        f1s.append(df['F1'].mean())

    data_dict = {'Model': models, 'Features': classes, 'Acc': accs, 'Nmi': nmis, 'Ari': aris, 'F1': f1s}
    data = pd.DataFrame(data_dict)

    directory = '../figures/pointSpectrum/feature_importance/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    custom_dict = {'full': 3, '-500': 2, '-1000': 1, '-2000': 0}
    data['Rank'] = data['Features'].map(custom_dict)
    data = data.sort_values(['Rank', 'Model'], ascending=False)

    for metric in ['Acc', 'Ari', 'Nmi', 'F1']:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4.8])
        custom_palette = sns.color_palette(colors)
        ax.grid(axis='y')
        sns.barplot(x='Features', y=metric, data=data, palette=custom_palette, hue='Model')
        plt.legend(loc='best', ncol=1, frameon=False)
        plt.tight_layout()
        plt.savefig('{}/features_{}.pdf'.format(directory, metric), format='pdf')
        plt.show()
# plot_features()