from rl.core import Agent
import keras.backend as K
from keras.optimizers import SGD
import numpy as np
import collections


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

class REINFORCE(Agent):
    def __init__(self, model, memory, gamma=1, batch_size=32, nb_steps_warmup=1000, *args, **kwargs):
        super(REINFORCE, self).__init__(*args, **kwargs)
        self.model = model
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.step_counter = 0
        self.compiled = False
        self.stack_trace = collections.deque()

    def print_config(self, optimizer_name, metrics, loss):
        print("[Info] AGENT CONFIGURATION")
        print("Optimizer: {}, Loss: {}, Metrics: {}".format(optimizer_name, loss, metrics))
        print("Memory Config: {}".format(self.memory.get_config()))
        print("Warmup steps: {}".format(self.nb_steps_warmup))
        print()

    def process_state_batch(self, batch):
        input_shape = (self.batch_size,) + self.model.input_shape[1:]
        batch = batch.reshape(input_shape)  # ASSUME FLATTEN OBSERVATION (25 for ekf, 2 for succruns)
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def process_reward_batch(self, rewards):
        batch = np.array(rewards)
        return batch.reshape((self.batch_size, 1))

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def q_eval_state(self, state):
        """
        Evaluation of the current state using QNetwork.
        :param state: state to be evaluated
        :return: Q value
        """
        input_shape = (self.batch_size,) + self.model.input_shape[1:]
        state_batch = state.reshape(input_shape)  # ASSUME FLATTEN OBSERVATION (25 for ekf, 2 for succruns)
        q_val = self.model.predict(state_batch)[0, 0]    # return only the value, no nasted struct
        return q_val

    def forward(self, observation):
        # There is no action
        self.stack_trace.append(observation)
        return 0    #No action, return always 0

    def backward(self, reward, terminal):
        if not terminal:
            return
        while self.stack_trace:
            observation = self.stack_trace.popleft()
            self.memory.append(observation, None, reward, False) #action is None, terminal never used
            self.step_counter = self.step_counter + 1
            if self.step_counter >= self.nb_steps_warmup:
                experience = self.memory.sample(1)[0]
                state_batch = self.process_state_batch(experience.state0[0])    #exp.state0 is list
                target_batch = self.process_reward_batch(experience.reward)    #here, introduce gamma
                #print("REWARD = {} | SAMPLE = {}".format(reward, target_batch[0]))
                self.model.train_on_batch(state_batch, target_batch)

    def compile(self, optimizer_name, metrics=[]):
        metrics += [mean_q]
        loss = 'mse'
        optimizer = optimizer_name
        if optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
        elif optimizer_name == 'sgdnestmom':
            optimizer = SGD(learning_rate=0.01, momentum=0.5, nesterov=True)
        self.model.compile(optimizer=optimizer, metrics=metrics, loss=loss)
        self.compiled = True
        self.print_config(optimizer_name, metrics, loss)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def layers(self):
        return self.model.layers[:]
