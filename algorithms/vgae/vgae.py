import numpy as np
import tensorflow as tf
from algorithms.vgae.models import VGAE, GAE
from utils.utils import preprocess_adj, preprocess_features, load_data, sparse_to_tuple, chebyshev_polynomials, save_results


def run_vgae(args):
    # with tf.device('/cpu:0'):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # load data
    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.input)
    print("Loaded", len(y_train), "nodes")
    print()
    print("-- Data format --")
    print("Adj:       ", A.shape, type(A), "number of indices", len(A.indices))
    print("y_train:   ", y_train.shape, "\t", type(y_train))
    print("train_mask:", train_mask.shape, "\t", type(train_mask))
    print("features:", features.shape, "\t", type(features))

    num_classes = y_train.shape[1]
    # D^-1@X
    features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]
    features = sparse_to_tuple(features)
    print('features coordinates::', features[0].shape)
    print('Non-zero feature entries::', features[1].shape)
    print('features shape::', features[2])

    if args.type == 'normal':
        A = [preprocess_adj(A)]
        A = sparse_to_tuple(A)
    elif args.type == 'cheby':
        A = chebyshev_polynomials(A, args.max_degree)
        A = sparse_to_tuple(A)
    else:
        return

    # init model and optimizer
    if args.model == 'vgae':
        model = VGAE(input_dim=features[2][1], output_dim=args.dimension, hidden_dim=args.hidden, num_features_nonzero=features[1].shape,
                     dropout=args.dropout, is_sparse_inputs=args.sparse)
    else:
        model = GAE(input_dim=features[2][1], output_dim=args.dimension, hidden_dim=args.hidden, num_features_nonzero=features[1].shape,
                    dropout=args.dropout, is_sparse_inputs=args.sparse)

    train_label = tf.convert_to_tensor(y_train)
    train_mask = tf.convert_to_tensor(train_mask)
    val_label = tf.convert_to_tensor(y_val)
    val_mask = tf.convert_to_tensor(val_mask)
    test_label = tf.convert_to_tensor(y_test)
    test_mask = tf.convert_to_tensor(test_mask)
    features = tf.SparseTensor(*features)
    support = [tf.cast(tf.SparseTensor(*A[0]), dtype=tf.float32)]

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(args.epochs):
        with tf.GradientTape() as tape:
            loss = model((features, train_mask, support), training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print('Epoch:', epoch, '\tloss:', float(loss))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            model.save_weights('output/models/best_{}'.format(args.model))
        else:
            cnt_wait += 1

        if cnt_wait == args.early_stopping:
            print('Early stopping!')
            break

    test_loss = model((features,  test_mask, support), training=False)

    print('test loss:', float(test_loss))

    # load model
    print('Loading {}th epoch'.format(best_t))
    model.load_weights('output/models/best_{}'.format(args.model))

    # produce embeddings
    embeds = model.embed((features, support))
    # save embeddings
    embeds = [e.numpy() for e in embeds]
    save_results(args, embeds)
