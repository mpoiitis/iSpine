from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from algorithms.dane.models import AutoEncoder


class PreTrainer(object):

    def __init__(self, config):
        self.config = config
        self.net_input_dim = config['net_input_dim']
        self.att_input_dim = config['att_input_dim']
        self.net_hidden = config['net_hidden']
        self.net_dimension = config['net_dimension']
        self.att_hidden = config['att_hidden']
        self.att_dimension = config['att_dimension']
        self.pretrain_params_path = config['pretrain_params_path']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.dropout = config['drop_prob']

        self.W_init = {}
        self.b_init = {}

    def pretrain(self, data, model):
        if model == 'net':
            input_shape = self.net_input_dim
            hidden_shape = self.net_hidden
            dimension_shape = self.net_dimension
        else:  #  model == 'att'
            input_shape = self.att_input_dim
            hidden_shape = self.att_hidden
            dimension_shape = self.att_dimension

        autoencoder = AutoEncoder(hidden1_dim=hidden_shape, hidden2_dim=dimension_shape, output_dim=input_shape, dropout=self.dropout)
        optimizer = Adam(lr=self.learning_rate)
        autoencoder.compile(optimizer=optimizer, loss=MeanSquaredError())
        autoencoder.fit(data, data, epochs=50, batch_size=self.batch_size, shuffle=True)

        autoencoder.save_weights(self.pretrain_params_path + '_' + model)

        return autoencoder
