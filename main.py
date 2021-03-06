from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from algorithms.gcn.gcn import run_gcn
from algorithms.dgi.dgi import run_dgi
from algorithms.vgae.vgae import run_vgae
from algorithms.agc.agc import run_agc
from algorithms.dane.dane import run_dane
from algorithms.gat.gat import run_gat
from algorithms.pointspectrum.pointspectrum import run_pointSpectrum
from algorithms.kspace_gnn.kspace_gnn import run_kspace_gnn
from algorithms.age.age import run_age


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

    point_spectrum_parser: ArgumentParser = embedding_subparsers.add_parser('pointSpectrum', help='pointSpectrum algorithm')
    point_spectrum_parser.add_argument("-d", "--dims", nargs='+', type=int, default=[100], help='Number of units in hidden layers for the autoencoder. Example --dims 500 200. Default is [100].')
    point_spectrum_parser.add_argument("-dr", "--dropout", default=0.2, type=float, help="Dropout rate (1 - keep probability). Default is 0.2.")
    point_spectrum_parser.add_argument("-lr", "--learning-rate", default=0.01, type=float, help="Initial learning rate. Default is 0.01.")
    point_spectrum_parser.add_argument("-e", "--epochs", default=500, type=int, help="Number of epochs. Default is 2000.")
    point_spectrum_parser.add_argument("-p", "--power", default=8, type=int, help="The upper bound of convolution order to search. Default is 8.")
    point_spectrum_parser.add_argument("-t", "--temperature", default=10, type=int, help="The degree to which the differentiable layer approximates KMeans clustering. Default is 10.")
    point_spectrum_parser.add_argument("-a", "--alpha", default=1, type=float, help="The reconstruction loss factor. Default is 1.")
    point_spectrum_parser.add_argument("-ap", "--a-prog", default='linear', type=str, choices=['linear', 'lineardec', 'exp', 'expdec', 'const'], help="How to calculate alpha for every training epoch. [linear, lineardec, exp, expdec, const]. Default is const")
    point_spectrum_parser.add_argument("-b", "--beta", default=1, type=float, help="The point loss factor. Default is 1.")
    point_spectrum_parser.add_argument("-bp", "--b-prog", default='linear', type=str, choices=['linear', 'lineardec', 'exp', 'expdec', 'const'], help="How to calculate beta for every training epoch. [linear, lineardec, exp, expdec, const]. Default is const")
    point_spectrum_parser.add_argument("-es", "--es", default=None, type=int, help="If given, it enables early stopping, tracking the given number of epochs.")
    point_spectrum_parser.add_argument("--save", dest='save', action='store_true', help="If given, it saves the embedding on disk.")

    kSpaceGnn_parser = embedding_subparsers.add_parser('kspace_gnn', help='kSPACE GNN algorithm')
    kSpaceGnn_parser.add_argument("-r", "--repeats", default=1, type=int, help="How many times to repeat the experiment. Default is 1.")
    kSpaceGnn_parser.add_argument("-d", "--dims", nargs='+', type=int, default=[32], help='Number of units in hidden layers. Example --dims 500 200. Default is [32].')
    kSpaceGnn_parser.add_argument("-dr", "--dropout", default=0.4, type=float, help="Dropout rate (1 - keep probability). Default is 0.4.")
    kSpaceGnn_parser.add_argument("-lr", "--learning-rate", default=0.005, type=float, help="Initial learning rate. Default is 0.001.")
    kSpaceGnn_parser.add_argument("-e", "--epochs", default=2000, type=int, help="Number of epochs. Default is 2000.")
    kSpaceGnn_parser.add_argument("-t", "--temperature", default=10, type=int, help="The degree to which the differentiable layer approximates KMeans clustering. Default is 10.")
    kSpaceGnn_parser.add_argument("-a-max", "--a-max", default=1, type=float, help="The upper bound of alpha rate. Default is 1.")
    kSpaceGnn_parser.add_argument("-a", "--alpha", default='linear', type=str, choices=['linear', 'exp', 'const'], help="How to calculate alpha for every training epoch. [linear, exp, const]. Default is const")
    kSpaceGnn_parser.add_argument("--save", dest='save', action='store_true', help="If given, it saves the results.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

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
    elif args.method == 'pointSpectrum':
        run_pointSpectrum(args)
    elif args.method == 'kspace_gnn':
        for i in range(args.repeats):
            run_kspace_gnn(args)
        # # from algorithms.kspace_gnn.utils import plot_metrics
        # # plot_metrics('input_cora-method_kspace_gnn-repeats_10-dims_500_500_250_200_100-dropout_0.2-learning_rate_0.001-epochs_500-p_epochs_100-a_max_1-slack_400-alpha_const-save_True')
        # from algorithms.kspace_gnn.utils import plot_metrics_for_layers
        # plot_metrics_for_layers()
    elif args.method == 'age':
        run_age(args)
    else:
        pass


