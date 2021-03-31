import time
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from .utils import get_alpha, calc_metrics, cluster_kl_loss
from .models import GAE, GCNEncoder


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

    args.dims = [dataset.num_features] + args.dims # add the initial dimension of features for the 1st encoder layer

    print_data_stats(dataset)

    data = dataset[0] # only one graph exists in these datasets
    data.train_mask = data.val_mask = data.test_mask = None # reset masks

    original_data = data.clone() # keep original data for the last evaluation step
    data = train_test_split_edges(data) # apart from the classic usage, it also creates positive edges (contained) and negative ones (not contained in graph)
    model = GAE(args.dims, GCNEncoder(args.dims, args.dropout))

    alphas = get_alpha(args.a_max, args.epochs, args.slack, args.alpha)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) # it uses the default decoder, which is the pair-wise similarity
    data = data.to(device)
    original_data = original_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # writer = SummaryWriter('logs/fit/kSpaceGNN_' + str(time.time()))
    for epoch in range(args.epochs):
        model.train()

        x = data.x
        train_pos_edge_index = data.train_pos_edge_index
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        rec_loss = model.recon_loss(z, train_pos_edge_index)

        if epoch >= args.slack - 1:
            c_loss = cluster_kl_loss(z, model.centers)
            # c_loss = tf.reduce_mean(cluster_q_loss(z, model.centers))
            correction_factor = abs(rec_loss / c_loss)
            c_loss = c_loss * correction_factor
            loss = rec_loss + alphas[epoch] * c_loss
        else:
            loss = rec_loss

        if epoch >= args.slack - 1 and epoch % 5:
            for param in model.parameters():
                param.requires_grad = False
            model.centers.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True
            model.centers.requires_grad = False

        loss.backward()
        optimizer.step()
        auc, ap = test(data, model)
        print('Epoch: {}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))

        # writer.add_scalar('auc_train', auc, epoch)
        # writer.add_scalar('ap_train', ap, epoch)

    print("Optimization Finished!")
    x = original_data.x
    edge_index = original_data.edge_index

    y = original_data.y.cpu().detach().numpy()
    z = model.encode(x, edge_index).cpu().detach()
    m = len(np.unique(y))

    kmeans = KMeans(n_clusters=m)
    pred = kmeans.fit_predict(z)
    acc, nmi, f1, ari = calc_metrics(pred, y)
    print('Acc= {:.4f}%    Nmi= {:.4f}%    Ari= {:.4f}%   Macro-f1= {:.4f}%'.format(acc * 100, nmi * 100, ari * 100, f1 * 100))