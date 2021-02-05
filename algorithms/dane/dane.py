from .utils.walks import *
from .utils.utils import generate_samples, sample_by_idx
from algorithms.dane.pretrainer import PreTrainer
from .models import FullModel
import os
import tensorflow as tf
from utils.utils import preprocess_features, load_data, save_results


def run_dane(args):
    """
           :param args: CLI arguments
           """
    if not os.path.exists('output/models'):
        os.makedirs('output/models')

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # load data
    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.input)

    num_nodes = A.shape[1]
    num_features = features.shape[1]
    num_classes = y_train.shape[1]

    # D^-1@X
    features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]

    walk_config = {
        'num_walks': args.num_walks,
        'walk_length': args.walk_length,
        'window_size': args.window_size,
    }

    walks = get_walks(A, walk_config)  # use walks instead of the original adjacency for pretraining

    if not os.path.exists('./Log/{}'.format(args.input)):
        os.makedirs('./Log/{}'.format(args.input))

    # PRE-TRAIN
    pretrain_config = {
        'net_hidden': args.net_hidden,
        'net_dimension': args.dimension,
        'att_hidden': args.att_hidden,
        'att_dimension': args.dimension,
        'net_input_dim': num_nodes,
        'att_input_dim': num_features,
        'drop_prob': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'pretrain_params_path': './Log/{}/pretrain_params'.format(args.input)}

    pretrainer = PreTrainer(pretrain_config)
    net_model = pretrainer.pretrain(walks, 'net')
    att_model = pretrainer.pretrain(features.todense(), 'att')

    # TRAIN
    # features = sparse_to_tuple(features)
    # A = sparse_to_tuple(coo_matrix(A))
    # walks = sparse_to_tuple(coo_matrix(walks))

    features = features.todense()
    A = A.todense()
    A = A.astype(np.float64)

    net_model.load_weights(pretrain_config['pretrain_params_path'] + '_net')
    att_model.load_weights(pretrain_config['pretrain_params_path'] + '_att')
    model_config = {
        'beta': args.beta,
        'alpha': args.alpha,
        'gamma': args.gamma
    }

    model = FullModel(net_model, att_model, model_config)
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

    best = 1e9
    best_t = 0
    cnt_wait = 0
    for epoch in range(args.epochs):

        idx1, idx2 = generate_samples(walks, features, A, model, args.batch_size, num_nodes)

        index = 0
        cost = 0.0
        cnt = 0
        while True:
            if index > num_nodes:
                break
            if index + args.batch_size < num_nodes:
                mini_batch1 = sample_by_idx(idx1[index:index + args.batch_size], walks, features, A)
                mini_batch2 = sample_by_idx(idx2[index:index + args.batch_size], walks, features, A)
            else:
                mini_batch1 = sample_by_idx(idx1[index:], walks, features, A)
                mini_batch2 = sample_by_idx(idx2[index:], walks, features, A)
            index += args.batch_size

            with tf.GradientTape() as tape:
                loss = model((mini_batch1.X, mini_batch2.X, mini_batch1.Z, mini_batch2.Z, mini_batch1.W, mini_batch2.W))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            cost += loss
            cnt += 1

        cost /= cnt
        print('Epoch:', epoch, '\tloss:', float(cost))

        if cost < best:
            best = cost
            best_t = epoch
            cnt_wait = 0
            model.save_weights('output/models/best_{}'.format(args.method))
        else:
            cnt_wait += 1

        if cnt_wait == args.early_stopping:
            print('Early stopping!')
            break

    # INFER
    # load model
    print('Loading {}th epoch'.format(best_t))
    model.load_weights('output/models/best_dane')

    # produce embeddings
    _, _, embeds = model.embed((walks, features))

    # save embeddings
    embeds = [e.numpy() for e in embeds]
    save_results(args, embeds)

