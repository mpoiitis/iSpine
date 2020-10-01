from .utils.walks import *
from .utils import gpu_info
from .dataset.dataset import Dataset
from .trainer.pretrainer import PreTrainer
from .trainer.trainer import Trainer
from .model.model import Model
import os
import tensorflow as tf

def run_dane(args):
    tf.compat.v1.disable_eager_execution()
    graph_config = {
        'is_adjlist': args.is_adjlist,
        'graph_file': 'data/dane_data/{}/edges.txt'.format(args.input),
        'label_file': args.label_file,
        'feature_file': args.feature_file,
        'node_status_file': args.node_status,
    }

    walk_config = {
        'num_walks': args.num_walks,
        'walk_length': args.walk_length,
        'window_size': args.window_size,
        'walks_file': args.output
    }

    graph = Graph(graph_config)
    get_walks(graph, walk_config)

    gpus_to_use, free_memory = gpu_info.get_free_gpu()
    print('GPU:', gpus_to_use, 'FREE MEMORY:', free_memory)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use  # 0

    random.seed(9001)

    dataset_config = {'feature_file': args.feature_file,
                      'graph_file': 'data/dane_data/{}/edges.txt'.format(args.input),
                      'walks_file': args.output,
                      'label_file': args.label_file}
    graph = Dataset(dataset_config)

    if not os.path.exists('./Log/{}'.format(args.input)):
        os.makedirs('./Log/{}'.format(args.input))

    pretrain_config = {
        'net_shape': args.net_shape,
        'att_shape': args.att_shape,
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'drop_prob': args.dropout,
        'pretrain_params_path': './Log/{}/pretrain_params.pkl'.format(args.input)}

    pretrainer = PreTrainer(pretrain_config)
    pretrainer.pretrain(graph.X, 'net')
    pretrainer.pretrain(graph.Z, 'att')

    model_config = {
        'net_shape': args.net_shape,
        'att_shape': args.att_shape,
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': './Log/{}/pretrain_params.pkl'.format(args.input)}

    model = Model(model_config)

    trainer_config = {
        'net_shape': args.net_shape,
        'att_shape': args.att_shape,
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'drop_prob': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'beta': args.beta,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'model_path': './Log/{}/{}_model.pkl'.format(args.input, args.input)}

    trainer = Trainer(model, trainer_config)
    trainer.train(graph)
    trainer.infer(graph)
