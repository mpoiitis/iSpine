import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_dense_adj, train_test_split_edges
import torch
import pickle
import time
import random
from sklearn.cluster import KMeans
from .models import PointSpectrum
from utils.utils import largest_eigval_smoothing_filter, preprocess_adj, get_file_count, get_factor, EarlyStopping
from utils.plots import plot_centers
from .utils import calc_metrics, write_to_csv

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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


def run_pointSpectrum(args):
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

    adj = to_dense_adj(data.edge_index)
    adj = adj.reshape(adj.shape[1], adj.shape[2])
    adj = sp.coo_matrix(adj.cpu().numpy())
    adj_normalized = preprocess_adj(adj)

    h = largest_eigval_smoothing_filter(adj_normalized)
    h_k = h ** args.power
    X = h_k.dot(data.x)

    data = train_test_split_edges(data)  # apart from the classic usage, it also creates positive edges (contained) and negative ones (not contained in graph)

    # CREATE MODEL
    model = PointSpectrum(dims, m, args.dropout, args.temperature)
    if args.es:
        early_stopping = EarlyStopping(patience=args.es)
    alphas = get_factor(args.alpha, args.epochs, args.a_prog)
    betas = get_factor(args.beta, args.epochs, args.b_prog)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    X = torch.FloatTensor(X).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    r_loss_list = []
    c_loss_list = []
    best_acc = 0
    best_nmi = 0
    best_ari = 0
    best_f1 = 0
    best_epoch = -1
    start = time.time()
    for epoch in range(args.epochs):
        train_pos_edge_index = data.train_pos_edge_index

        optimizer.zero_grad()
        z = model.encode(X)

        loss, r_loss, c_loss = model.loss(z, alphas[epoch], betas[epoch], train_pos_edge_index)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            z = model.encode(X)
            pred = model.assign_clusters(z).cpu().detach().numpy()
            # z_cpu = z.cpu().detach()
            # kmeans = KMeans(n_clusters=m)
            # pred = kmeans.fit_predict(z_cpu)
            acc, nmi, ari, f1 = calc_metrics(pred, y)

        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)
        r_loss_list.append(r_loss)
        c_loss_list.append(c_loss)
        print('Epoch: {}, Loss: {:.4f}, Rec: {:.4f}, Clust: {:.4f}, Acc= {:.4f}%    Nmi= {:.4f}%    Ari= {:.4f}%   Macro-f1= {:.4f}%'.format(epoch, loss, r_loss, c_loss, acc * 100, nmi * 100, ari * 100, f1 * 100))

        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1

        # if args.save:
        #     if (epoch != 0 and epoch % 500 == 0) or epoch == (args.epochs - 1):
        #         model.eval()
        #         z = model.encode(X)
        #
        #         centers = model.mu.cpu().detach()
        #         plot_centers(z, centers, y, args, epoch)
        #         model.train()
        if args.es:
            early_stopping(loss)
            if early_stopping.early_stop:
                break
    total_time = time.time() - start
    print("Total time: ", total_time)
    print("Optimization Finished!")
    model.eval()
    z = model.encode(X)

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
        directory = 'pickles/{}/{}_a_{}_a_fun_{}_b _{}_b_fun_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims_{}_power'.format(args.method, args.alpha, args.a_prog, args.beta, args.b_prog, args.temperature, args.epochs, args.learning_rate, args.dropout, dims, args.power)
        file_count = get_file_count(directory, 'reclosses')
        torch.save(model.state_dict(), '{}/model'.format(directory))
        pickle.dump(X, open('{}/X.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(acc_list, open('{}/accs_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(nmi_list, open('{}/nmis_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(ari_list, open('{}/aris_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(f1_list, open('{}/f1s_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(r_loss_list, open('{}/reclosses_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(c_loss_list, open('{}/clustlosses_{}.pickle'.format(directory, file_count), 'wb'))

    write_to_csv(args, best_epoch, best_acc, best_nmi, best_ari, best_f1, total_time)