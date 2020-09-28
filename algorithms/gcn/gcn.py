import os
import numpy as np
import tensorflow as tf
from utils.utils import preprocess_adj, preprocess_features, sparse_to_tuple, chebyshev_polynomials, load_data, save_results
from algorithms.gcn.models import GCN


def run_gcn(args):
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

    if args.type == 'gcn':
        A = [preprocess_adj(A)]
        A = sparse_to_tuple(A)
    elif args.type == 'gcn_cheby':
        A = chebyshev_polynomials(A, args.max_degree)
        A = sparse_to_tuple(A)
    else:
        return

    model = GCN(input_dim=features[2][1], output_dim=num_classes, hidden_dim=args.dimension, num_features_nonzero=features[1].shape,
                dropout=args.dropout, weight_decay=args.weight_decay)

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
            loss, acc = model((features, train_label, train_mask, support), training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        _, val_acc = model((features, val_label, val_mask, support), training=False)


        print('Epoch:', epoch, '\tloss:', float(loss), '\ttrain:', float(acc), '\tval:', float(val_acc))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            model.save_weights('output/models/best_gcn')
        else:
            cnt_wait += 1

        if cnt_wait == args.early_stopping:
            print('Early stopping!')
            break

    test_loss, test_acc = model((features, test_label, test_mask, support), training=False)

    print('test loss:', float(test_loss), '\ttest acc:', float(test_acc))

    # load model
    print('Loading {}th epoch'.format(best_t))
    model.load_weights('output/models/best_gcn')

    # produce embeddings
    embeds = model.embed((features, support), training=False)
    # save embeddings
    embeds = [e.numpy() for e in embeds]
    save_results(args, embeds)
