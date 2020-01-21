import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, PReLU, Activation
from tensorflow.keras import Model


class EKFModel(Model):
    def __init__(self, model_params=dict()):
        self.manage_params(model_params)
        assert 'batch_size' in self.model_params.keys()
        assert 'hidden_initializer' in self.model_params.keys()
        assert 'hidden_activation' in self.model_params.keys()
        assert 'last_activation' in self.model_params.keys()

        super(EKFModel, self).__init__()
        self.state_variables = 9
        self.batch_size = self.model_params['batch_size']
        self.d1 = Dense(128, input_dim=(self.batch_size, self.state_variables), activation=self.get_activation(self.model_params['hidden_activation']))
        self.d2 = Dense(64, activation=self.get_activation(self.model_params['hidden_activation']))
        self.d3 = Dense(16, activation=self.get_activation(self.model_params['hidden_activation']))
        self.out = Dense(1, activation=self.get_activation(self.model_params['last_activation']))

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.out(x)

    def print_config(self):
        print("[Info] Model (NN) Configuration")
        print("[Info] Batch Size: {}".format(self.model_params['batch_size']))
        print("[Info] Hidden Initializer: {}".format(self.model_params['hidden_initializer']))
        print("[Info] Hidden Activation Function: {}".format(self.model_params['hidden_activation']))
        print("[Info] Last Activation Function: {}".format(self.model_params['last_activation']))
        print()

    def manage_params(self, model_params):
        self.model_params = self.get_default_params_EKF()
        if len(model_params.keys()) > 0:
          self.update_params(model_params)

    def get_default_params_EKF(self):
        params = dict()
        params['batch_size'] = 32
        params['hidden_initializer'] = 'glorot_uniform'
        params['hidden_activation'] = 'relu'
        params['last_activation'] = 'linear'
        return params

    def update_params(self, params):
        param_names = ['batch_size', 'hidden_initializer', 'hidden_activation', 'last_activation']
        for name in param_names:
            if name in params.keys():
                self.model_params[name] = params[name]

    def get_activation(self, name):
        if name == 'leakyrelu':
            return LeakyReLU(alpha=0.3)
        elif name == 'prelu':
            return PReLU()
        else:
            return Activation(name)