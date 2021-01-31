import os
import sys
import warnings
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from .models import LinTrans
from .optimizer import loss_function
from sklearn.cluster import SpectralClustering, KMeans
from utils.metrics import clustering
from utils.utils import load_data_trunc, largest_eigval_smoothing_filter
from tqdm import tqdm
from sklearn.preprocessing import normalize

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def update_similarity(z, upper_threshold, lower_treshold):
    f_adj = np.matmul(z, np.transpose(z))
    cosine = f_adj
    cosine = cosine.reshape([-1, ])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)


def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth


def run_age(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda is True:
        print('Using GPU')
        torch.cuda.manual_seed(SEED)
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    adj, features, gnd, idx_train, idx_val, idx_test = load_data_trunc(args.dataset)
    n_nodes, feat_dim = features.shape
    dims = [feat_dim] + args.dims

    # convert one hot labels to integer ones
    if args.input != "wiki":
        gnd = np.argmax(gnd, axis=1)
        features = features.todense()
    features = features.astype(np.float32)

    m = len(np.unique(gnd))  # number of clusters according to ground truth
    # Cluster = SpectralClustering(n_clusters=m, affinity='precomputed', random_state=0)
    Cluster = KMeans(n_clusters=m)

    # Store original adjacency matrix (without diagonal entries) for later
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    sm_fea_s = sp.csr_matrix(features).toarray()

    print('Laplacian Smoothing...')
    h = largest_eigval_smoothing_filter(adj)
    h_k = h ** args.gnnlayers
    X = h_k.dot(sm_fea_s)


    db, _, _, _, _, _ = clustering(Cluster, X, gnd, age=True)
    best_cl = db

    model = LinTrans(args.linlayers, dims)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    X = torch.FloatTensor(X)

    if args.cuda:
        model.cuda()
        inx = X.cuda()
    else:
        inx = X

    up_eta = (args.upth_ed - args.upth_st) / (args.epochs / args.upd)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs / args.upd)
    pos_inds, neg_inds = update_similarity(normalize(X.numpy()), args.upth_st, args.lowth_st)
    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)

    bs = min(args.bs, len(pos_inds))
    if args.cuda:
        pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
    else:
        pos_inds_cuda = torch.LongTensor(pos_inds)
    print('Start Training...')
    for epoch in tqdm(range(args.epochs)):

        st, ed = 0, bs
        batch_num = 0
        model.train()
        length = len(pos_inds)

        while ed <= length:
            if args.cuda:
                sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed - st)).cuda()
            else:
                sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed - st))
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            t = time.time()
            optimizer.zero_grad()
            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            x = torch.index_select(inx, 0, xind)
            y = torch.index_select(inx, 0, yind)
            zx = model(x)
            zy = model(y)
            if args.cuda:
                batch_label = torch.cat((torch.ones(ed - st), torch.zeros(ed - st))).cuda()
            else:
                batch_label = torch.cat((torch.ones(ed - st), torch.zeros(ed - st)))
            batch_pred = model.dcs(zx, zy)
            loss = loss_function(adj_preds=batch_pred, adj_labels=batch_label, n_nodes=ed - st)

            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            st = ed
            batch_num += 1
            if ed < length <= ed + bs:
                ed += length - ed
            else:
                ed += bs

        if (epoch + 1) % args.upd == 0:
            model.eval()
            mu = model(inx)
            hidden_emb = mu.cpu().data.numpy()
            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth)
            bs = min(args.bs, len(pos_inds))
            if args.cuda:
                pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
            else:
                pos_inds_cuda = torch.LongTensor(pos_inds)

            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, time.time() - t))

            db, acc, nmi, adjscore = clustering(Cluster, hidden_emb, gnd, age=True)
            tqdm.write("DB: {} ACC: {} NMI: {} ARI: {}".format(db, acc, nmi, adjscore))
            if db >= best_cl:
                best_cl = db
                best_acc = acc
                best_nmi = nmi
                best_adj = adjscore

    tqdm.write("Optimization Finished!")
    tqdm.write('best_acc: {}, best_nmi: {}, best_ari: {}'.format(best_acc, best_nmi, best_adj))
