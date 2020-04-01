import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU, PReLU, Activation

class ModelFactory:
    def __init__(self, problem_name, observation_space_shape, num_actions, model_params=dict()):
        assert problem_name in ['EKF', 'SR']
        assert num_actions == 1         # Since Policy Evaluation -> only 1 output (state value)
        self.manage_params(problem_name, model_params)
        assert 'hidden_initializer' in self.model_params.keys()
        assert 'hidden_activation' in self.model_params.keys()
        assert 'last_activation' in self.model_params.keys()
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        if problem_name == 'EKF':
            self.model = self.get_model_EKF()
        elif problem_name == 'SR':
            self.model = self.get_model_SR()
        self.print_config()

    def print_config(self):
        print("[Info] MODEL (NN) CONFIGURATION")
        print(self.model.summary())
        print("Hidden Initializer: {}".format(self.model_params['hidden_initializer']))
        print("Hidden Activation Function: {}".format(self.model_params['hidden_activation']))
        print("Last Activation Function: {}".format(self.model_params['last_activation']))
        print()

    def manage_params(self, problem_name, model_params):
        self.set_default_params(problem_name)
        if len(model_params.keys()) > 0:
            self.update_params(model_params)

    def set_default_params(self, problem_name):
        # problem name might be "" if the parameters are all defined by user input
        if problem_name == 'EKF':
            self.model_params = self.get_default_params_EKF()
        elif problem_name == 'SR':
            self.model_params = self.get_default_params_SR()
        else:
            self.model_params = dict()

    def get_default_params_EKF(self):
        params = dict()
        params['hidden_initializer'] = 'glorot_uniform'
        params['hidden_activation'] = 'relu'
        params['last_activation'] = 'linear'
        return params

    def get_default_params_SR(self):
        params = dict()
        params['hidden_initializer'] = 'glorot_uniform'
        params['hidden_activation'] = 'leakyrelu'
        params['last_activation'] = 'linear'
        return params

    def update_params(self, params):
        param_names = ['hidden_initializer', 'hidden_activation', 'last_activation']
        for name in param_names:
            if name in params.keys():
                self.model_params[name] = params[name]

    def get_model(self):
        return self.model

    def get_model_EKF(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.observation_space_shape))
        model.add(Dense(128, kernel_initializer=self.model_params['hidden_initializer']))
        model.add(self.get_activation(self.model_params['hidden_activation']))
        model.add(Dense(64, kernel_initializer=self.model_params['hidden_initializer']))
        model.add(self.get_activation(self.model_params['hidden_activation']))
        model.add(Dense(16, kernel_initializer=self.model_params['hidden_initializer']))
        model.add(self.get_activation(self.model_params['hidden_activation']))
        model.add(Dense(self.num_actions))
        model.add(self.get_activation(self.model_params['last_activation']))
        return model

    def get_model_SR(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.observation_space_shape))
        model.add(Dense(16))
        model.add(self.get_activation(self.model_params['hidden_activation']))
        model.add(Dense(8))
        model.add(self.get_activation(self.model_params['hidden_activation']))
        model.add(Dense(self.num_actions))
        model.add(self.get_activation(self.model_params['last_activation']))
        return model

    def get_activation(self, name):
        if name == 'leakyrelu':
            return LeakyReLU(alpha=0.3)
        elif name == 'prelu':
            return PReLU()
        else:
            return Activation(name)
