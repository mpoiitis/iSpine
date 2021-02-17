from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from algorithms.gcn.gcn import run_gcn
from algorithms.dgi.dgi import run_dgi
from algorithms.vgae.vgae import run_vgae
from algorithms.agc.agc import run_agc
from algorithms.dane.dane import run_dane
from algorithms.kspace.kspace import run_kspace
from algorithms.gat.gat import run_gat
from algorithms.age.age import run_age

from utils.plots import plot_results


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('input', choices=['cora', 'citeseer', 'pubmed', 'wiki'], help="Input graph dataset. Options: ['cora', 'citeseer', 'pubmed', 'wiki']")

    embedding_subparsers = parser.add_subparsers(dest='method', help="Specifies whether to run an embedding generation algorithm or an embedding evaluation method.")

    gcn_parser = embedding_subparsers.add_parser('gcn', help='GCN method.')
    gcn_parser.add_argument("--type", default='gcn', type=str, choices=['gcn', 'gcn_cheby'], help="Type of adjacency matrix. Default is gcn.")
    gcn_parser.add_argument("--epochs", default=200, type=int, help="Number of epochs. Default is 200.")
    gcn_parser.add_argument("--dimension", default=128, type=int, help="Number of latent dimensions to learn for each node. Default is 128.")
    gcn_parser.add_argument("--batch-size", default=128, type=int, help="Size of the batch used for training. Default is 128.")
    gcn_parser.add_argument("--max-degree", default=3, type=int, help="Maximum Chebyshev polynomial degree. Default is 3.")
    gcn_parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate. Default is 0.001.")
    gcn_parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate (1 - keep probability). Default is 0.5.")
    gcn_parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight for L2 loss on embedding matrix. E.g. 0.008. Default is 0.0.")
    gcn_parser.add_argument("--early-stopping", default=None, type=int, help="Tolerance for early stopping (# of epochs). E.g. 10. Default is None.")

    dgi_parser = embedding_subparsers.add_parser('dgi', help='DGI method.')
    dgi_parser.add_argument("--epochs", default=200, type=int, help="Number of epochs. Default is 200.")
    dgi_parser.add_argument("--dimension", default=512, type=int,  help="Number of latent dimensions to learn for each node. Default is 512.")
    dgi_parser.add_argument("--batch-size", default=1, type=int, help="Size of the batch used for training. Default is 1.")
    dgi_parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate. Default is 0.001.")
    dgi_parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate (1 - keep probability). Default is 0.0.")
    dgi_parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight for L2 loss on embedding matrix. E.g. 0.008. Default is 0.0.")
    dgi_parser.add_argument("--early-stopping", default=20, type=int, help="Tolerance for early stopping (# of epochs). E.g. 10. Default is 20.")
    dgi_parser.add_argument("--sparse", dest='sparse', action='store_true', help="Use sparse form of arrays")

    vgae_parser = embedding_subparsers.add_parser('vgae', help='VGAE method.')
    vgae_parser.add_argument("--type", default='normal', type=str, choices=['normal', 'cheby'], help="Type of adjacency matrix. Default is normal.")
    vgae_parser.add_argument("--model", default='gae', type=str, choices=['gae', 'vgae'], help="Type of model. Default is gae.")
    vgae_parser.add_argument("--epochs", default=200, type=int, help="Number of epochs. Default is 200.")
    vgae_parser.add_argument("--iter", default=15, type=int, help="Number of iterations for the whole process. Default is 15.")
    vgae_parser.add_argument("--batch-size", default=1, type=int, help="Size of the batch used for training. Default is 1.")
    vgae_parser.add_argument("--hidden", default=512, type=int, help="Neurons in hidden layer. Default is 512.")
    vgae_parser.add_argument("--dimension", default=128, type=int, help="Embedding dimension. Default is 128.")
    vgae_parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate. Default is 0.001.")
    vgae_parser.add_argument("--early-stopping", default=20, type=int, help="Tolerance for early stopping (# of epochs). E.g. 10. Default is 20.")
    vgae_parser.add_argument("--sparse", dest='sparse', action='store_true', help="If given, use sparse form of arrays")
    vgae_parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate (1 - keep probability). Default is 0.0.")

    agc_parser = embedding_subparsers.add_parser('agc', help='AGC method.')
    agc_parser.add_argument("--max-iter", default=60, type=int, help="Number of max iterations if there is no conversion in intra_C. Default is 60.")

    dane_parser = embedding_subparsers.add_parser('dane', help='DANE method.')
    dane_parser.add_argument('--num-walks', default=10, type=int, help="Number of walks. Default is 10.")
    dane_parser.add_argument('--walk-length', default=80, type=int, help="The length of the walk. Default is 80.")
    dane_parser.add_argument('--window-size', default=10, type=int, help="The size of the window. Default is 10.")
    dane_parser.add_argument("--dimension", default=100, type=int, help="Embedding dimension. Default is 100.")
    dane_parser.add_argument('--net-hidden', default=200, type=int, help="Hidden layer dimension of connectivity network. Default is 200.")
    dane_parser.add_argument('--att-hidden', default=200, type=int, help="Hidden layer dimension of attribute network. Default is 200.")
    dane_parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate (1 - keep probability). Default is 0.2.")
    dane_parser.add_argument("--learning-rate", default=0.00001, type=float, help="Initial learning rate. Default is 0.00001.")
    dane_parser.add_argument("--batch-size", default=100, type=int, help="Size of the batch used for training. Default is 100.")
    dane_parser.add_argument("--epochs", default=500, type=int, help="Number of epochs. Default is 500.")
    dane_parser.add_argument("--early-stopping", default=20, type=int, help="Tolerance for early stopping (# of epochs). E.g. 10. Default is 20.")
    dane_parser.add_argument("--alpha", default=50., type=float, help="Initial learning rate. Default is 50.")
    dane_parser.add_argument("--beta", default=100., type=float, help="Initial learning rate. Default is 100.")
    dane_parser.add_argument("--gamma", default=500., type=float, help="Initial learning rate. Default is 500.")

    gat_parser = embedding_subparsers.add_parser('gat', help='GAT method.')
    gat_parser.add_argument("--epochs", default=200, type=int, help="Number of epochs. Default is 200.")
    gat_parser.add_argument("--batch-size", default=1, type=int, help="Size of the batch used for training. Default is 1.")
    gat_parser.add_argument("--hidden", default=8, type=int, help="Neurons in hidden layer. Default is 8.")
    gat_parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate. Default is 0.001.")
    gat_parser.add_argument("--early-stopping", default=20, type=int, help="Tolerance for early stopping (# of epochs). E.g. 10. Default is 20.")
    gat_parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight for L2 loss on embedding matrix. E.g. 0.008. Default is 0.0.")
    gat_parser.add_argument("--sparse", dest='sparse', action='store_true', help="Use sparse form of arrays")
    gat_parser.add_argument("--residual", dest='residual', action='store_true', help="If given, determines whether to add seq to ret")
    gat_parser.add_argument("--ffd-drop", default=0.6, type=float, help="Dropout rate (1 - keep probability) for features. Default is 0.6.")
    gat_parser.add_argument("--attn-drop", default=0.6, type=float, help="Dropout rate (1 - keep probability) for attention. Default is 0.6.")

    age_parser = embedding_subparsers.add_parser('age', help='AGE method.')
    age_parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers. Default is 3")
    age_parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden linear layers. Default is 1")
    age_parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train. Default is 400')
    age_parser.add_argument('--dims', nargs='+', type=int, default=[500], help='Number of units in hidden layers. Default is [500]')
    age_parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate. Default is 0.001')
    age_parser.add_argument('--upth_st', type=float, default=0.0015, help='Upper Threshold start. Default is 0.0015')
    age_parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start. Default is 0.1')
    age_parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end. Default is 0.001')
    age_parser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold end. Default is 0.5')
    age_parser.add_argument('--upd', type=int, default=10, help='Update epoch. Default is 10. Default is 10')
    age_parser.add_argument('--bs', type=int, default=10000, help='Batchsize. Default is 10000')
    age_parser.add_argument('--dataset', type=str, default='citeseer', help='Name of dataset. Used for saving results. Default is citeseer')
    age_parser.add_argument('--no-cuda', action='store_true', default=False, help='If given, disables CUDA training.')

    kSpace_parser = embedding_subparsers.add_parser('kspace', help='kSPACE algorithm')
    kSpace_parser.add_argument("--repeats", default=1, type=int, help="How many times to repeat the experiment. Default is 1.")
    kSpace_parser.add_argument("--model", default='ae', type=str, choices=['ae', 'vae', 'dae', 'dvae'], help="Type of autoencoder. Simple, variational or denoising. Default is ae.")
    kSpace_parser.add_argument('--dims', nargs='+', type=int, default=[200, 100], help='Number of units in hidden layers. Default is [200, 100]. Example --dims 500 200')
    kSpace_parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate (1 - keep probability). Default is 0.2.")
    kSpace_parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate. Default is 0.001.")
    kSpace_parser.add_argument("--batch-size", default=100, type=int, help="Size of the batch used for training. Default is 100.")
    kSpace_parser.add_argument("--epochs", default=500, type=int, help="Number of epochs. Default is 500.")
    kSpace_parser.add_argument("--early-stopping", default=20, type=int, help="Tolerance for early stopping (# of epochs). E.g. 10. Default is 20.")
    kSpace_parser.add_argument("--power", default=8, type=int, help="The upper bound of convolution order to search. Default is 8.")
    kSpace_parser.add_argument("--a-max", default=5, type=float, help="The upper bound of alpha rate. Default is 5.")
    kSpace_parser.add_argument("--alpha", default='linear', type=str, choices=['linear', 'exp', 'zeros'], help="How to calculate alpha for every training epoch. [linear, exp]. Default is linear")
    kSpace_parser.add_argument("--save", dest='save', action='store_true', help="If given, it saves the embedding on disk.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # config = {'Dataset': 'pubmed', 'Model': 'ae', 'Dimension': 100, 'Power': 7,
    #           'Epochs': 500, 'Batch Size': 100, 'Learning Rate': 0.001, 'a': 'linear', 'a_max': 5, 'Dropout': 0.2}
    # plot_results(config, 'Hidden')

    args = parse_args()
    if args.method == 'gcn':
        run_gcn(args)
    elif args.method == 'vgae':
        run_vgae(args)
    elif args.method == 'dgi':
        run_dgi(args)
    elif args.method == 'agc':
        run_agc(args)
    elif args.method == 'dane':
        run_dane(args)
    elif args.method == 'gat':
        run_gat(args)
    elif args.method == 'kspace':
        run_kspace(args)
        # from algorithms.kspace.kspace import run_kspace_grid_search
        # run_kspace_grid_search()
    elif args.method == 'age':
        run_age(args)
    else:
        pass


