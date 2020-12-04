from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def tsne(embeds, gnd, args):
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("bright", len(np.unique(gnd)))
    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(embeds)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], ax=ax1, hue=gnd, legend='full', palette=palette)
    ax1.set_title('T-SNE {}'.format(args.input))

    plt.savefig('figures/mymethod/tsne/{}_{}epochs_{}dims_{}hidden.png'.format(args.input, args.epochs, args.dimension,
                                                                               args.hidden), format='png')
    plt.show()