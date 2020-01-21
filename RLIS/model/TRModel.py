import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, PReLU, Activation
from tensorflow.keras import Model


class EKFModel(Model):
    def __init__(self, batch_size=32, hidden_initializer='glorot_uniform', hiddent_activation='relu', last_activation='linear'):
        super(EKFModel, self).__init__()
        self.state_variables = 9
        self.batch_size = batch_size
        self.hidden_init = hidden_initializer
        self.hidden_act = hiddent_activation
        self.last_act = last_activation

        self.d1 = Dense(128, input_dim=(self.batch_size, self.state_variables),
                        kernel_initializer=self.hidden_init, activation=self.get_activation(self.hidden_act))
        self.d2 = Dense(64, kernel_initializer=self.hidden_init, activation=self.get_activation(self.hidden_act))
        self.d3 = Dense(16, kernel_initializer=self.hidden_init, activation=self.get_activation(self.hidden_act))
        self.out = Dense(1, activation=self.get_activation(self.last_act))

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.out(x)

    def print_config(self):
        print("[Info] Model (NN) Configuration")
        print("[Info] Batch Size: {}".format(self.batch_size))
        print("[Info] Hidden Initializer: {}".format(self.hidden_init))
        print("[Info] Hidden Activation Function: {}".format(self.hidden_act))
        print("[Info] Last Activation Function: {}".format(self.last_act))
        print()

    def get_activation(self, name):
        if name == 'leakyrelu':
            return LeakyReLU(alpha=0.3)
        elif name == 'prelu':
            return PReLU()
        else:
            return Activation(name)