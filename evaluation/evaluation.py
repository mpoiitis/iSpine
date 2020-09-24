import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, f1_score


def evaluation(args):
    if args.no_embedding is True:
        data = pd.read_csv(args.ground_truth_input, sep="\t", header=None)
        labels = data.iloc[:, -1]

        # turn labels from string into integers
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

        column_names = list(data.columns)
        data = data.iloc[:, 1:-1]  # drop ids and labels
    else:
        data = pd.read_csv(args.input)
        gt = pd.read_csv(args.ground_truth_input, sep="\t", header=None)
        labels = gt.iloc[:, -1]
        # turn labels from string into integers
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

        column_names = list(data.columns)
        data = data.drop(['id'], axis=1)

    if args.normalize:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=column_names[1:])

    kmeans = KMeans(init='k-means++', n_clusters=args.k, random_state=0)
    kmeans.fit(data)
    y_pred = kmeans.predict(data)
    silhouette = silhouette_score(data, y_pred)
    print('Silhouette score: ',  silhouette)
    if args.k == len(set(labels)):
        f1 = f1_score(labels, y_pred, average='macro')
        print('F1 score: ', f1)