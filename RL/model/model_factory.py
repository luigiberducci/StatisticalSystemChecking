import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten

class ModelFactory:
    def __init__(self, problem_name, observation_space_shape, num_actions):
        assert problem_name in ['EKF', 'SR']
        assert num_actions == 1         # Since Policy Evaluation -> only 1 output (state value)
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        if problem_name == 'EKF':
            self.model = self.get_model_EKF()
        elif problem_name == 'SR':
            self.model = self.get_model_SR()
        print(self.model.summary())

    def get_model(self):
        return self.model

    def get_model_EKF(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.observation_space_shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        return model

    def get_model_SR(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.observation_space_shape))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        return model