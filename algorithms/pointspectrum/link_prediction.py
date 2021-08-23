import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_dense_adj, train_test_split_edges
import torch
import pickle
import time
import random
import os
from .models import PointSpectrum
from utils.utils import largest_eigval_smoothing_filter, preprocess_adj, get_file_count, get_factor, EarlyStopping, mask_test_edges
from .utils import write_to_csv_link, get_roc_score

# Enable this if you want to get reproducible results
# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)


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

    data = dataset[0]  # only one graph exists in these datasets
    data.train_mask = data.val_mask = data.test_mask = None  # reset masks

    original_data = data.clone()  # keep original data for the last evaluation step
    y = original_data.y.cpu()
    m = len(np.unique(y))  # number of clusters

    adj = to_dense_adj(data.edge_index)
    adj = adj.reshape(adj.shape[1], adj.shape[2])

    adj_orig = adj

    adj = sp.coo_matrix(adj.cpu().numpy())
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    src = []
    dst = []
    for edge in train_edges:
        src.append(edge[0])
        dst.append(edge[1])
    train_pos_edge_index = [src, dst]

    adj = adj_train

    x_dir = 'pickles/X/'
    if not os.path.exists(x_dir):
        os.makedirs(x_dir)
    x_pickle = x_dir + 'X_{}_{}.pickle'.format(args.power, args.input)
    if os.path.exists(x_pickle):
        print('loading X...')
        X = pickle.load(open(x_pickle, 'rb'))
    else:
        adj_normalized = preprocess_adj(adj)

        h = largest_eigval_smoothing_filter(adj_normalized)
        h_k = h ** args.power
        X = h_k.dot(data.x)

        pickle.dump(X, open('{}'.format(x_pickle), 'wb'))


    # CREATE MODEL
    model = PointSpectrum(dims, m, args.dropout, args.temperature, enc_type=args.enc)

    if args.es:
        early_stopping = EarlyStopping(patience=args.es)
    alphas = get_factor(args.alpha, args.epochs, args.a_prog)
    betas = get_factor(args.beta, args.epochs, args.b_prog)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X = torch.FloatTensor(X).to(device)
    train_pos_edge_index = torch.LongTensor(np.array(train_pos_edge_index)).to(device)

    dims = '_'.join([str(v) for v in args.dims])
    directory = 'pickles/{}/link/{}_{}_{}_a_{}_a_fun_{}_b _{}_b_fun_{}_temperature_{}_epochs_{}_lr_{}_dropout_{}_dims_{}_power'.format(args.method, args.enc, args.input, args.alpha, args.a_prog, args.beta, args.b_prog, args.temperature, args.epochs, args.learning_rate, args.dropout, dims, args.power)
    if os.path.exists('{}/model'.format(directory)):
        print('loading model...')
        model.load_state_dict(torch.load('{}/model'.format(directory)))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        auc_list = []
        ap_list = []
        loss_list = []
        r_loss_list = []
        c_loss_list = []
        best_auc = 0
        best_ap = 0
        best_epoch = -1
        start = time.time()
        for epoch in range(args.epochs):

            optimizer.zero_grad()
            z = model(X)

            loss, r_loss, c_loss = model.loss(z, alphas[epoch], betas[epoch], train_pos_edge_index)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                z = model(X)
                auc, ap = get_roc_score(z.cpu().detach(), adj_orig, val_edges, val_edges_false)

            auc_list.append(auc)
            ap_list.append(ap)
            loss_list.append(loss)
            r_loss_list.append(r_loss)
            c_loss_list.append(c_loss)
            print('Epoch: {}, Loss: {:.4f}, Rec: {:.4f}, Clust: {:.4f}, Auc= {:.4f}%    AP= {:.4f}% '.format(epoch, loss, r_loss, c_loss, auc * 100, ap * 100))

            if auc > best_auc:
                best_epoch = epoch
                best_auc = auc
                best_ap = ap
                if args.save:
                    torch.save(model.state_dict(), '{}/model'.format(directory))

            if args.es:
                early_stopping(loss)
                if early_stopping.early_stop:
                    break
        total_time = time.time() - start
        print("Total time: ", total_time)
        print("Optimization Finished!")
        print('Best    Epoch= {}    Auc= {:.4f}%    AP= {:.4f}%'.format(best_epoch, best_auc * 100, best_ap * 100))

    model.eval()
    z = model(X)
    auc, ap = get_roc_score(z.cpu().detach(), adj_orig, val_edges, val_edges_false)
    print('Model    Auc= {:.4f}%    AP= {:.4f}%'.format(auc * 100, ap * 100))

    if args.save:
        file_count = get_file_count(directory, 'reclosses')
        pickle.dump(auc_list, open('{}/aucs_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(ap_list, open('{}/aps_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(loss_list, open('{}/losses_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(r_loss_list, open('{}/reclosses_{}.pickle'.format(directory, file_count), 'wb'))
        pickle.dump(c_loss_list, open('{}/clustlosses_{}.pickle'.format(directory, file_count), 'wb'))

    write_to_csv_link(args, best_epoch, best_auc, best_ap, total_time)
