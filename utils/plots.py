from sklearn.manifold import TSNE
from textwrap import wrap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import io

def tsne(embeds, gnd, args):
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("bright", len(np.unique(gnd)))
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(embeds)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], ax=ax1, hue=gnd, legend='full', palette=palette)
    ax1.set_title('T-SNE {}'.format(args.input))

    plt.savefig('figures/kspace/tsne/{}_{}epochs_{}dims_{}hidden.png'.format(args.input, args.epochs, args.dimension,
                                                                               args.hidden), format='png')
    plt.show()


def plot_centers(embeds, centers, gnd, epoch):
    figure = plt.figure(figsize=(11.7, 8.27))
    palette = sns.color_palette("bright", len(np.unique(gnd)))
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(embeds)
    centers_embedded = tsne.fit_transform(centers)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=gnd, legend='full', palette=palette)
    plt.scatter(centers_embedded[:, 0], centers_embedded[:, 1], c='red', marker='X')
    plt.title('T-SNE embeds and centers, epoch: {}'.format(epoch))
    plt.tight_layout()
    plt.show()


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


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
