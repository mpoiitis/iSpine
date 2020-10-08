import tensorflow as tf
import numpy as np
from utils.utils import load_data, preprocess_features, preprocess_adj_bias, adj_to_bias, save_results
from .models import GAT
from evaluation.evaluation import auto_kmeans


def train(model, inputs, bias_mat, lbl_in, msk_in, optimizer):
    with tf.GradientTape() as tape:
        logits, accuracy, loss = model(inputs=inputs, training=True, bias_mat=bias_mat, lbl_in=lbl_in, msk_in=msk_in)

    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(gradient_variables)

    return logits, accuracy, loss


def evaluate(model, inputs, bias_mat, lbl_in, msk_in):
    logits, accuracy, loss = model(inputs=inputs, bias_mat=bias_mat, lbl_in=lbl_in, msk_in=msk_in, training=False)
    return logits, accuracy, loss


def run_gat(args):

    hid_units = args.hidden  # numbers of hidden units per each attention head in each layer
    n_heads = [hid_units, 1]  # additional entry for the output layer

    # load data
    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.input)
    features = preprocess_features(features).todense()

    num_nodes = features.shape[0]
    num_classes = y_train.shape[1]

    features = features[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    if args.sparse:
        biases = preprocess_adj_bias(A)
    else:
        A = A.todense()
        A = A[np.newaxis]
        biases = adj_to_bias(A, [num_nodes], nhood=1)

    model = GAT(hid_units, n_heads, num_classes, num_nodes, args.sparse, ffd_drop=args.ffd_drop, attn_drop=args.attn_drop,
                activation=tf.nn.elu, residual=args.residual)

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

    min_loss = np.inf
    max_acc = 0.0
    curr_step = 0

    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # TRAINING
        step = 0
        train_size = features.shape[0]
        while step * args.batch_size < train_size:

            if args.sparse:
                bbias = biases
            else:
                bbias = biases[step * args.batch_size:(step + 1) * args.batch_size]

            _, acc, loss = train(model, inputs=features[step * args.batch_size:(step + 1) * args.batch_size],
                                             bias_mat=bbias,
                                             lbl_in=y_train[step * args.batch_size:(step + 1) * args.batch_size],
                                             msk_in=train_mask[step * args.batch_size:(step + 1) * args.batch_size],
                                             optimizer=optimizer)
            train_loss += loss
            train_acc += acc
            step += 1

        # VALIDATION
        step = 0
        val_size = features.shape[0]
        while step * args.batch_size < val_size:

            if args.sparse:
                bbias = biases
            else:
                bbias = biases[step * args.batch_size:(step + 1) * args.batch_size]

            _, acc, loss = evaluate(model, inputs=features[step * args.batch_size:(step + 1) * args.batch_size],
                                                bias_mat=bbias,
                                                lbl_in=y_val[step * args.batch_size:(step + 1) * args.batch_size],
                                                msk_in=val_mask[step * args.batch_size:(step + 1) * args.batch_size])
            val_loss += loss
            val_acc += acc
            step += 1

        print('Epoch %d     Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' % (epoch, train_loss / step, train_acc / step, val_loss / step, val_acc / step))

        # EARLY STOPPING

        if val_acc / step >= max_acc or val_loss / step <= min_loss:
            if val_acc / step >= max_acc and val_loss / step <= min_loss:
                vacc_early_model = val_acc / step
                vlss_early_model = val_loss / step
                working_weights = model.get_weights()
            max_acc = np.max((val_acc / step, max_acc))
            min_loss = np.min((val_loss / step, min_loss))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == args.early_stopping:
                print('Early stop! Min loss: ', min_loss, ', Max accuracy: ', max_acc)
                print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                model.set_weights(working_weights)
                break

    # TESTING

    step = 0
    test_size = features.shape[0]
    test_loss = 0.0
    test_acc = 0.0
    while step * args.batch_size < test_size:

        if args.sparse:
            bbias = biases
        else:
            bbias = biases[step * args.batch_size:(step + 1) * args.batch_size]

        _, acc, loss = evaluate(model, inputs=features[step * args.batch_size:(step + 1) * args.batch_size],
                                bias_mat=bbias, lbl_in=y_test[step * args.batch_size:(step + 1) * args.batch_size],
                                msk_in=test_mask[step * args.batch_size:(step + 1) * args.batch_size])
        test_loss += loss
        test_acc += acc
        step += 1

    print('Test loss:', test_loss / step, '; Test accuracy:', test_acc / step)