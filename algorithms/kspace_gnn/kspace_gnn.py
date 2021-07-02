import numpy as np
import scipy.sparse as sp
import torch
import pickle
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_dense_adj, train_test_split_edges
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from .utils import get_alpha, calc_metrics
from .models import ClusterGAE
from utils.plots import plot_centers
from utils.utils import get_file_count, largest_eigval_smoothing_filter, preprocess_adj


def test(data, model):
    model.eval()

    with torch.no_grad():
        x = data.x
        train_pos_edge_index = data.train_pos_edge_index
        test_pos_edge_index = data.test_pos_edge_index
        test_neg_edge_index = data.test_neg_edge_index

        z = model.encode(x, train_pos_edge_index)
        acc, f1 = model.test(z, test_pos_edge_index, test_neg_edge_index)

        return acc, f1


def print_data_stats(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]  # Get the first graph object.
    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def filter_graph(data, power):

    adj = to_dense_adj(data.edge_index)
    adj = adj.reshape(adj.shape[1], adj.shape[2])
    adj = sp.coo_matrix(adj.cpu().numpy())
    adj_normalized = preprocess_adj(adj)

    h = largest_eigval_smoothing_filter(adj_normalized)
    h_k = h ** power
    X = h_k.dot(data.x)

    return X


def get_similarity(data, power=8):
    X = filter_graph(data, power)
    S = 1 - pairwise_distances(X, metric="cosine")

    return S


def run_kspace_gnn(args):

    if args.input == 'cora':
        dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    elif args.input == 'citeseer':
        dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
    elif args.input == 'pubmed':
        dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
    else:
        print('Wikipedia dataset currently not supported!')
        return

    dims = [dataset.num_features] + args.dims  # add the initial dimension of features for the 1st encoder layer

    print_data_stats(dataset)

    data = dataset[0]  # only one graph exists in these datasets
    data.train_mask = data.val_mask = data.test_mask = None  # reset masks

    original_data = data.clone()  # keep original data for the last evaluation step
    y = original_data.y.cpu()
    m = len(np.unique(y))  # number of clusters

    X = get_similarity(data, 8)

    data = train_test_split_edges(data)  # apart from the classic usage, it also creates positive edges (contained) and negative ones (not contained in graph)

    model = ClusterGAE(dims, m, args.dropout, args.temperature)
    alphas = get_alpha(args.a_max, args.epochs, args.alpha)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    original_data = original_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    rec_loss_list = []
    c_loss_list = []
    best_acc = 0
    best_nmi = 0
    best_ari = 0
    best_f1 = 0
    best_epoch = -1
    for epoch in range(args.epochs):

        x = data.x
        train_pos_edge_index = data.train_pos_edge_index
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)

        loss, rec_loss, c_loss = model.complex_loss(z, alphas[epoch], train_pos_edge_index)
        loss.backward()
        optimizer.step()

        pred = model.assign_clusters(z).cpu().detach().numpy()
        acc, nmi, ari, f1 = calc_metrics(pred, y)
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)
        rec_loss_list.append(rec_loss)
        c_loss_list.append(c_loss)
        print('Epoch: {}, Loss: {:.4f}, Reconstruction: {:.4f}, Clustering: {:.4f}, Acc= {:.4f}%    Nmi= {:.4f}%    Ari= {:.4f}%   Macro-f1= {:.4f}%'.format(epoch, loss, rec_loss, c_loss, acc * 100, nmi * 100, ari * 100, f1 * 100))

        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
        # if args.save:
        #     if (epoch != 0 and epoch % 500 == 0) or epoch == (args.epochs - 1):
        #         model.eval()
        #         z = model.encode(original_data.x, original_data.edge_index).cpu().detach().numpy()
        #         centers = model.mu.cpu().detach()
        #         plot_centers(z, centers, y, args, epoch)
        #         model.train()

    print("Optimization Finished!")
    x = original_data.x
    edge_index = original_data.edge_index
    z = model.encode(x, edge_index)

    z_cpu = z.cpu().detach()
    kmeans = KMeans(n_clusters=m)
    pred = kmeans.fit_predict(z_cpu)
    acc, nmi, ari, f1 = calc_metrics(pred, y)
    print('KMeans   Acc= {:.4f}%    Nmi= {:.4f}%    Ari= {:.4f}%   Macro-f1= {:.4f}%'.format(acc * 100, nmi * 100, ari * 100, f1 * 100))
    pred = model.assign_clusters(z).cpu().detach().numpy()
    acc, nmi, ari, f1 = calc_metrics(pred, y)
    print('Model    Acc= {:.4f}%    Nmi= {:.4f}%    Ari= {:.4f}%   Macro-f1= {:.4f}%'.format(acc * 100, nmi * 100, ari * 100, f1 * 100))
    print('Best    Epoch= {}    Acc= {:.4f}%    Nmi= {:.4f}%    Ari= {:.4f}%   Macro-f1= {:.4f}%'.format(best_epoch, best_acc * 100, best_nmi * 100, best_ari * 100, best_f1 * 100))
    if args.save:

        dims = '_'.join([str(v) for v in args.dims])
        directory = 'pickles/{}/{}_a_{}_a-max_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims'.format(args.method, args.alpha, args.a_max, args.temperature, args.epochs, args.learning_rate, args.dropout, dims)
        file_count = get_file_count(directory, 'reclosses')
        pickle.dump(acc_list, open('{}/accs_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(nmi_list, open('{}/nmis_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(ari_list, open('{}/aris_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(f1_list, open('{}/f1s_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(rec_loss_list, open('{}/reclosses_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(c_loss_list, open('{}/clustlosses_{}.pickle'.format(directory, file_count), 'wb'))
