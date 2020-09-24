from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from evaluation.evaluation import evaluation
from algorithms.gcn.gcn import run_gcn
from algorithms.dgi.dgi import run_dgi


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--input', required=True, choices=['cora', 'citeseer', 'pubmed'], help="Input graph dataset. Options: ['cora', 'citeseer', 'pubmed']")
    parser.add_argument('--output', required=True, help='Output representation file path.')
    subparsers = parser.add_subparsers(dest='method', help="Specifies whether to run an embedding generation algorithm or an embedding evaluation method.")

    gcn_parser = subparsers.add_parser('gcn', help='GCN method.')
    gcn_parser.add_argument("--type", default='gcn', type=str, choices=['gcn', 'gcn_cheby'], help="Type of adjacency matrix. Default is gcn.")
    gcn_parser.add_argument("--epochs", default=200, type=int, help="Number of epochs. Default is 200.")
    gcn_parser.add_argument("--dimension", default=128, type=int, help="Number of latent dimensions to learn for each node. Default is 128.")
    gcn_parser.add_argument("--batch-size", default=128, type=int, help="Size of the batch used for training. Default is 128.")
    gcn_parser.add_argument("--max-degree", default=3, type=int, help="Maximum Chebyshev polynomial degree. Default is 3.")
    gcn_parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate. Default is 0.001.")
    gcn_parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate (1 - keep probability). Default is 0.5.")
    gcn_parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight for L2 loss on embedding matrix. E.g. 0.008. Default is 0.0.")
    gcn_parser.add_argument("--early-stopping", default=None, type=int, help="Tolerance for early stopping (# of epochs). E.g. 10. Default is None.")

    dgi_parser = subparsers.add_parser('dgi', help='DGI method.')
    dgi_parser.add_argument("--epochs", default=200, type=int, help="Number of epochs. Default is 200.")
    dgi_parser.add_argument("--dimension", default=512, type=int,  help="Number of latent dimensions to learn for each node. Default is 512.")
    dgi_parser.add_argument("--batch-size", default=1, type=int, help="Size of the batch used for training. Default is 1.")
    dgi_parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate. Default is 0.001.")
    dgi_parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate (1 - keep probability). Default is 0.0.")
    dgi_parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight for L2 loss on embedding matrix. E.g. 0.008. Default is 0.0.")
    dgi_parser.add_argument("--early-stopping", default=20, type=int, help="Tolerance for early stopping (# of epochs). E.g. 10. Default is 20.")
    dgi_parser.add_argument("--sparse", dest='sparse', action='store_true', help="Use sparse form of arrays")

    args = parser.parse_args()

    return args


def run_vae(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    if args.method == 'gcn':
        run_gcn(args)
    elif args.method == 'vae':
        pass
    elif args.method == 'dgi':
        run_dgi(args)
    else:
        pass

    # evaluation(args)

