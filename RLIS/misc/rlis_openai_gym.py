import gym
import numpy as np
import random as rnd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Create dense network
def dense_nn(input_size, layer_units):
    nn = Sequential()
    for i, units in enumerate(layer_units):
        if i == 0:
            nn.add(Dense(units, activation="relu", input_dim=input_size))
        elif i == len(layer_units)-1:
            nn.add(Dense(units, activation="linear"))
        else:
            nn.add(Dense(units, activation="relu"))
    nn.compile(loss=tf.keras.losses.MSE,
               optimizer=Adam(learning_rate=.01))
    return nn

def random_agent(n_episodes):
    return_list = []
    for episode in range(n_episodes):
        print("[Info] episode: {}/{}, ".format(episode, n_episodes), end="")

        reward_list = []
        env.reset()
        done = False
        while not done:
            act = env.action_space.sample()
            _, rew, done, _ = env.step(act)
            reward_list.append(rew)
        # compute return
        ret = 0
        for i, reward in enumerate(reward_list):
            if i == 0:
                ret += reward
            else:
                ret += gamma ** i * reward
        return_list.append(ret)
        print("return value: {}".format(ret))
    return return_list

def train_qlearning_agent(n_episodes):
    MIN_REPLAY_MEMORY_SIZE = 200
    MAX_REPLAY_MEMORY_SIZE = 100000
    epsilon = 0.05
    batch_size = 32

    qnet = dense_nn(env.observation_space.shape[0], [64, 128, 64, 2])
    replay = deque(maxlen=MAX_REPLAY_MEMORY_SIZE)
    return_list = []
    for episode in range(n_episodes):
        print("[Info] episode: {}/{}, ".format(episode, n_episodes), end="")

        obs = env.reset()
        done = False

        reward_list = []
        while not done:
            if rnd.random() < epsilon:
                act = env.action_space.sample()
            else:
                act = np.argmax(qnet(obs.reshape(1, obs.shape[0]))[0])
            new_obs, rew, done, info = env.step(act)
            replay.append((obs, act, rew, new_obs, done))
            reward_list.append(rew)
            obs = new_obs

            if len(replay) >= MIN_REPLAY_MEMORY_SIZE:
                batch_transitions = rnd.sample(replay, batch_size)

                X, y = [], []
                for t, (current_state, action, reward, next_state, done) in enumerate(batch_transitions):
                    new_qval = reward
                    if not done:
                        new_qval += gamma * np.max(qnet(next_state.reshape(1, next_state.shape[0]))[0])
                    current_qval = np.array(qnet(current_state.reshape(1, current_state.shape[0]))[0])
                    current_qval[action] = new_qval  # update qval for the given action
                    # prepare training data
                    X.append(current_state)
                    y.append(current_qval)

                qnet.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0)
        # compute return
        ret = 0
        for i, reward in enumerate(reward_list):
            if i == 0:
                ret += reward
            else:
                ret += gamma ** i * reward
        return_list.append(ret)
        print("return value: {}".format(ret))
    return qnet, return_list

def train_double_qlearning_agent(n_episodes):
    MIN_REPLAY_MEMORY_SIZE = 200
    MAX_REPLAY_MEMORY_SIZE = 100000
    epsilon = 0.05
    batch_size = 32
    n_episodes_update_target = 5

    qnet = dense_nn(env.observation_space.shape[0], [64, 128, 64, 2])
    targetnet = dense_nn(env.observation_space.shape[0], [64, 128, 64, 2])
    targetnet.set_weights(qnet.get_weights())

    replay = deque(maxlen=MAX_REPLAY_MEMORY_SIZE)
    return_list = []
    for episode in range(n_episodes):
        print("[Info] episode: {}/{}, ".format(episode, n_episodes), end="")

        obs = env.reset()
        done = False
        # update target network
        if episode % n_episodes_update_target == 0:
            targetnet.set_weights(qnet.get_weights())
        reward_list = []
        while not done:
            if rnd.random() < epsilon:
                act = env.action_space.sample()
            else:
                act = np.argmax(qnet(obs.reshape(1, obs.shape[0]))[0])
            new_obs, rew, done, info = env.step(act)
            replay.append((obs, act, rew, new_obs, done))
            reward_list.append(rew)
            obs = new_obs

            if len(replay) >= MIN_REPLAY_MEMORY_SIZE:
                batch_transitions = rnd.sample(replay, batch_size)

                X, y = [], []
                for t, (current_state, action, reward, next_state, done) in enumerate(batch_transitions):
                    new_qval = reward
                    if not done:
                        new_qval += gamma * np.max(targetnet (next_state.reshape(1, next_state.shape[0]))[0])
                    current_qval = np.array(qnet(current_state.reshape(1, current_state.shape[0]))[0])
                    current_qval[action] = new_qval  # update qval for the given action
                    # prepare training data
                    X.append(current_state)
                    y.append(current_qval)

                qnet.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0)
        # compute return
        ret = 0
        for i, reward in enumerate(reward_list):
            if i == 0:
                ret += reward
            else:
                ret += gamma ** i * reward
        return_list.append(ret)
        print("return value: {}".format(ret))
    return qnet, return_list

env = gym.make("CartPole-v1")
n_episodes = 20000
gamma = 1

random_returns = random_agent(n_episodes)
q_agent, q_returns = train_qlearning_agent(n_episodes)
double_q_agent, double_q_returns = train_double_qlearning_agent(n_episodes)

print("[Info] Random Agent: avg return: {:.3f}".format(sum(random_returns)/n_episodes))
print("[Info] Deep QLearning Agent: avg return: {:.3f}".format(sum(q_returns)/n_episodes))
print("[Info] Double QLearning Agent: avg return: {:.3f}".format(sum(double_q_returns)/n_episodes))

plt.plot(range(n_episodes), random_returns, label="Random Agent")
plt.plot(range(n_episodes), q_returns, label="Deep Q-Learning")
plt.plot(range(n_episodes), double_q_returns, label="Double Q-Learning")
plt.xlabel("Episodes")
plt.ylabel("Return Value")
plt.legend()
plt.show()

env.close()