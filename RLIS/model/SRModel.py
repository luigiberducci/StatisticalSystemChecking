import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, PReLU, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
import numpy as np


class SRModel(Model):
    def __init__(self, batch_size=32, hidden_initializer='glorot_uniform', hiddent_activation='relu', last_activation='linear', input_state_vars=None):
        super(SRModel, self).__init__()
        # State representation
        self.state_variables = 2
        self.state_filter = self.get_state_filter(input_state_vars)
        self.input_state_vars = np.where(self.state_filter)[0].shape[0]     # consistency with other model implementations
        # Model params
        self.batch_size = batch_size
        self.hidden_init = hidden_initializer
        self.hidden_act = hiddent_activation
        self.last_act = last_activation
        # Model architecture
        #self.d1 = Dense(16, input_shape=(self.batch_size, self.input_state_vars),
        #                kernel_initializer=self.hidden_init, activation=self.get_activation(self.hidden_act))
        #self.d2 = Dense(8, kernel_initializer=self.hidden_init, activation=self.get_activation(self.hidden_act))
        #self.out = Dense(1, activation=self.get_activation(self.last_act))

        self.model = Sequential()
        self.model.add(Dense(16, input_shape=(self.input_state_vars,), batch_size=self.batch_size))
        self.model.add(self.get_activation(self.hidden_act))
        self.model.add(Dense(8))
        self.model.add(self.get_activation(self.hidden_act))
        self.model.add(Dense(1))
        self.model.add(self.get_activation(self.last_act))


    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)

    def get_state_filter(self, num_state_vars):
        import ipdb
        ipdb.set_trace()
        state_filter = [False] * self.state_variables
        if num_state_vars is not None:
            if num_state_vars > self.state_variables or num_state_vars == 0:
                raise ValueError("num state variables {} not valid. SR has {} state variables.".format(num_state_vars, self.state_variables))
            elif num_state_vars==self.state_variables:
                state_filter = [True] * self.state_variables
            else:
                state_filter[:num_state_vars-1] = True
        else:
            state_filter = [True] * self.state_variables
        return state_filter

    def get_model(self):
        return self.model

    def print_config(self):
        print("[Info] Model (NN) Configuration")
        print("[Info] Batch Size: {}".format(self.batch_size))
        print("[Info] Hidden Initializer: {}".format(self.hidden_init))
        print("[Info] Hidden Activation Function: {}".format(self.hidden_act))
        print("[Info] Last Activation Function: {}".format(self.last_act))
        print("[Info] Input State Representation: size: {}, filter: {}".format(self.input_state_vars, self.state_filter))
        print()

    def get_activation(self, name):
        if name == 'leakyrelu':
            return LeakyReLU(alpha=0.3)
        elif name == 'prelu':
            return PReLU()
        else:
            return Activation(name)
