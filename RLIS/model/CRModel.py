import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, PReLU, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
import numpy as np


class CRModel(Model):
    def __init__(self, batch_size=32, hidden_initializer='glorot_uniform', hiddent_activation='relu', last_activation='linear'):
        super(CRModel, self).__init__()
        # State representation
        self.state_variables = 13   # x, y, theta, v, xx, yy, ttheta, vv, cov4x4flat, time, realv, realu
        self.state_filter = [True] * self.state_variables
        self.input_state_vars = np.where(self.state_filter)[0].shape[0]     # input variables (reduced)
        # Model params
        self.batch_size = batch_size
        self.hidden_init = hidden_initializer
        self.hidden_act = hiddent_activation
        self.last_act = last_activation
        # Model architecture
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(self.input_state_vars,), batch_size=self.batch_size))
        self.model.add(self.get_activation(self.hidden_act))
        self.model.add(Dense(64))
        self.model.add(self.get_activation(self.hidden_act))
        self.model.add(Dense(16))
        self.model.add(self.get_activation(self.hidden_act))
        self.model.add(Dense(1))
        self.model.add(self.get_activation(self.last_act))

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
        print("[Info] Input State Representation: size: {}, filter: {}".format(self.input_state_vars, self.state_filter))
        print()

    def get_model(self):
        return self.model

    def get_activation(self, name):
        if name == 'leakyrelu':
            return LeakyReLU(alpha=0.3)
        elif name == 'prelu':
            return PReLU()
        else:
            return Activation(name)
