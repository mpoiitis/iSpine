import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, f1_score
import operator


def auto_kmeans(args=None, embedding_file=None, gt_file=None, kmax=10, normalize=False):
    embedding_file = args.input if args else embedding_file
    gt_file = args.gt_input if args else gt_file
    normalize = args.normalize if args else normalize
    kmax = args.kmax if args else kmax

    data = pd.read_csv(embedding_file)
    if gt_file:
        gt = pd.read_csv(gt_file, sep="\t", header=None)
        labels = gt.iloc[:, -1]
        # turn labels from string into integers
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

    column_names = list(data.columns)
    data = data.drop(['id'], axis=1)

    if normalize:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=column_names[1:])

    sil = dict()  # contains pairs {k: silhouette score}
    assignments = dict()  # contains pairs {k: cluster_assignments}
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax + 1):
        kmeans = KMeans(init='k-means++', n_clusters=k, random_state=0).fit(data)
        y_pred = kmeans.predict(data)
        silhouette = silhouette_score(data, y_pred)
        assignments.update({k: y_pred})
        sil.update({k: silhouette})
        print('K:', k, ' Silhouette score:',  silhouette)
        if gt_file:
            if k == len(set(labels)):
                f1 = f1_score(labels, y_pred, average='macro')
                print('F1 score: ', f1)
    best_k = max(sil.items(), key=operator.itemgetter(1))[0]
    print('Best k:', best_k)

    return assignments[best_k]
