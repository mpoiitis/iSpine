import tensorflow as tf
from .layers import Inference
from utils.metrics import masked_softmax_cross_entropy, masked_accuracy


class GAT(tf.keras.Model):
    def __init__(self, hid_units, n_heads, nb_classes, nb_nodes, sparse, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, l2_coef=0.0, residual=False):
        super(GAT, self).__init__()
        '''
        hid_units: This is the number of hidden units per each attention head in each layer (8). Array of hidden layer dimensions
        n_heads: This is the additional entry of the output layer [8,1]. More specifically the output that calculates attn
        nb_classes: This refers to the number of classes (7)
        nb_nodes: This refers to the number of nodes (2708)    
        activation: This is the activation function tf.nn.elu
        residual: This determines whether we add seq to ret (False)
        '''
        self.hid_units = hid_units  # [8]
        self.n_heads = n_heads  # [8,1]
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.l2_coef = l2_coef
        self.residual = residual

        self.inferencing = Inference(n_heads, hid_units, nb_classes, nb_nodes, sparse=sparse, ffd_drop=ffd_drop,
                                     attn_drop=attn_drop, activation=activation, residual=residual)

    def __call__(self, inputs, training, bias_mat, lbl_in, msk_in):
        logits = self.inferencing(inputs=inputs, bias_mat=bias_mat, training=training)

        log_resh = tf.reshape(logits, [-1, self.nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])

        loss = masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)

        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.l2_coef

        loss = loss + lossL2
        accuracy = masked_accuracy(log_resh, lab_resh, msk_resh)

        return logits, accuracy, loss