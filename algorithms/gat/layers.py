import tensorflow as tf


class Attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes=None, in_drop=0.0, coef_drop=0.0, activation=tf.nn.elu, residual=False):
        super(Attn_head, self).__init__()
        self.activation = activation
        self.residual = residual

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)

        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim, 1, use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):

        seq = self.in_dropout(seq, training=training)
        seq_fts = self.conv_no_bias(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        coefs = self.coef_dropout(coefs, training=training)
        seq_fts = self.in_dropout(seq_fts, training=training)

        vals = tf.matmul(coefs, seq_fts)
        vals = tf.cast(vals, dtype=tf.float32)
        ret = vals + self.bias_zero

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        return self.activation(ret)


class Sp_attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes, in_drop=0.0, coef_drop=0.0, activation=tf.nn.elu, residual=False):
        super(Sp_attn_head, self).__init__()
        self.hidden_dim = hidden_dim
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)

        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim, 1, use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):

        adj_mat = bias_mat

        seq = self.in_dropout(seq, training=training)
        seq_fts = self.conv_no_bias(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)

        f_1 = tf.reshape(f_1, (self.nb_nodes, 1))
        f_2 = tf.reshape(f_2, (self.nb_nodes, 1))
        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.compat.v1.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.compat.v2.sparse.softmax(lrelu)

        if training != False:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=self.coef_dropout(coefs.values, training=training),
                                    dense_shape=coefs.dense_shape)
            seq_fts = self.in_dropout(seq_fts, training=training)

        coefs = tf.compat.v2.sparse.reshape(coefs, [self.nb_nodes, self.nb_nodes])

        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse.sparse_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, self.nb_nodes, self.hidden_dim])

        ret = vals + self.bias_zero

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        return self.activation(ret)


def choose_attn_head(sparse):
    if sparse:
        chosen_attention = Sp_attn_head
    else:
        chosen_attention = Attn_head

    return chosen_attention


class Inference(tf.keras.layers.Layer):
    def __init__(self, n_heads, hid_units, nb_classes, nb_nodes, sparse, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(Inference, self).__init__()

        attned_head = choose_attn_head(sparse)

        self.attns = []
        self.sec_attns = []
        self.final_attns = []
        self.final_sum = n_heads[-1]

        for i in range(n_heads[0]):
            self.attns.append(attned_head(hidden_dim=hid_units[0], nb_nodes=nb_nodes,
                                          in_drop=ffd_drop, coef_drop=attn_drop,
                                          activation=activation,
                                          residual=residual))

        for i in range(1, len(hid_units)):
            sec_attns = []
            for j in range(n_heads[i]):
                sec_attns.append(attned_head(hidden_dim=hid_units[i], nb_nodes=nb_nodes,
                                             in_drop=ffd_drop, coef_drop=attn_drop,
                                             activation=activation,
                                             residual=residual))
                self.sec_attns.append(sec_attns)

        for i in range(n_heads[-1]):
            self.final_attns.append(attned_head(hidden_dim=nb_classes, nb_nodes=nb_nodes,
                                                in_drop=ffd_drop, coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual))

    def __call__(self, inputs, bias_mat, training):
        first_attn = []
        out = []

        for indiv_attn in self.attns:
            first_attn.append(indiv_attn(seq=inputs, bias_mat=bias_mat, training=training))
        h_1 = tf.concat(first_attn, axis=-1)

        for sec_attns in self.sec_attns:
            next_attn = []
            for indiv_attn in sec_attns:
                next_attn.append(indiv_attn(seq=h_1, bias_mat=bias_mat, training=training))
            h_1 = tf.concat(next_attn, axis=-1)

        for indiv_attn in self.final_attns:
            out.append(indiv_attn(seq=h_1, bias_mat=bias_mat, training=training))
        logits = tf.add_n(out) / self.final_sum

        return logits