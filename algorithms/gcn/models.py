from tensorflow.keras import Model
from .layers import GraphConvolution
import tensorflow as tf
from utils.metrics import masked_softmax_cross_entropy, masked_accuracy


class GCN(Model):

    def __init__(self, input_dim, output_dim, hidden_dim, num_features_nonzero, dropout, weight_decay, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input_dim = input_dim  # 1433
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.weight_decay = weight_decay

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        self.layers_ = []
        self.layers_.append(GraphConvolution(input_dim=self.input_dim,  # 1433
                                            output_dim=self.hidden_dim,  # 16
                                            num_features_nonzero=num_features_nonzero,
                                            activation=tf.nn.relu,
                                            dropout=self.dropout,
                                            is_sparse_inputs=True))





        self.layers_.append(GraphConvolution(input_dim=self.hidden_dim,
                                            output_dim=self.output_dim,
                                            num_features_nonzero=num_features_nonzero,
                                            activation=lambda x: x,
                                            dropout=self.dropout))


    def call(self, inputs, training=None):

        x, label, mask, support = inputs

        outputs = [x]

        for layer in self.layers:
            hidden = layer((outputs[-1], support), training)
            outputs.append(hidden)
        output = outputs[-1]

        # Weight decay loss
        loss = tf.zeros([])
        for var in self.layers_[0].trainable_variables:
            loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        loss += masked_softmax_cross_entropy(output, label, mask)

        acc = masked_accuracy(output, label, mask)

        return loss, acc

    def embed(self, inputs, training=False):
        x, support = inputs
        outputs = [x]
        for i, layer in enumerate(self.layers):
            # skip last layer as it does the annotation to the labels instead of the embedding
            if i == len(self.layers) - 1:
                break
            hidden = layer((outputs[-1], support), training)
            outputs.append(hidden)
        output = outputs[-1]

        return output

    def predict(self):
        return tf.nn.softmax(self.outputs)