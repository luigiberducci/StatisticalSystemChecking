import numpy as np
import gym
import gym_ekf_localization
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import Callback
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

ENV_NAME = 'ekf-loc-v0'

class EpsDecayCallback(Callback):
    """
        Implementation of epsilon decay as a Callback for EpsGreedyPolicy.
    """
    def __init__(self, eps_policy, decay_rate=0.95, min_eps=0.1):
        """
        :param eps_policy:  every policy based on an epsilon parameter
        :param decay_rate:  factor which reduce the current epsilon
        :param min_eps:     minimum value of epsilon, after reached stop decay
        """
        self.eps_policy = eps_policy
        self.decay_rate = decay_rate
        self.min_eps = min_eps
    def on_episode_begin(self, episode, logs={}):
        if self.eps_policy.eps > self.min_eps:
            self.eps_policy.eps *= self.decay_rate
        #print("NEW EPSILON: {}".format(self.eps_policy.eps))

def build_model(observation_space_shape, num_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + observation_space_shape))
    import ipdb
    ipdb.set_trace()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    print(model.summary())
    return model

def build_agent(obs_space_shape, num_actions):
    # Policy
    MAX_EPSILON = 1.0       # Start Exploration with high `epsilon` (random move factor)
    MIN_EPSILON = 0.1       # No lower than this minimum value
    DECAY_NUM_STEPS = 10000 # Decay from max to min in this number of steps
    VALUE_TEST = 0.05       # ?
    # Experience replay
    WARMUP_STEPS = 500      # Collect the first steps before start experience replay
    MEM_LIMIT = 5000        # Max number of steps to store
    MEM_WINDOW_LEN = 1      # Experience of lenght 1 (single step)
    # Target network
    TARGET_MODEL_UPD_RATE = 1e-2    # Update target network with this rate
    # Build network, exp. replay and policy
    model = build_model(obs_space_shape, num_actions)
    replay = SequentialMemory(limit=MEM_LIMIT, window_length=MEM_WINDOW_LEN)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=MAX_EPSILON, value_min=MIN_EPSILON,
                                  value_test=VALUE_TEST, nb_steps=DECAY_NUM_STEPS)
    # Finally build the agent
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=replay, nb_steps_warmup=WARMUP_STEPS,
                   target_model_update=TARGET_MODEL_UPD_RATE, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    callbacks = None  #not used for now
    return dqn, callbacks

def test_agent(agent=None):
    """DEPRECATED"""
    print("[TEST] Run UNTRAINED AGENT in 1 episode until reach a success (error detected).")
    input("Continue?")
    env = gym.make(ENV_NAME)
    num_actions = env.action_space.n
    obs_space_shape = env.observation_space.shape
    if agent is None:
        print("[Info] Building new agent")
        agent, callbacks = build_agent(obs_space_shape, num_actions)
    agent.test(env, nb_episodes=1, visualize=True)
    env.holdon_plot()

def test_environment():
    """DEPRECATED"""
    env = gym.make(ENV_NAME)
    observation = env.reset()
    for t in range(10):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("Rew {}, IsDone {}".format(reward, done))
        if done:
            print("Finished after {} steps".format(t+1))

def estimate_frequency_via_mc(number_of_observation):
    env = gym.make(ENV_NAME)
    counter = 0
    for i in range(number_of_observation):
        observation = env.reset()
        if env.is_done:
            counter = counter + 1
    return counter

def estimate_frequency_via_agent(number_of_observation, agent, visualize=False):
    env = gym.make(ENV_NAME)
    MAX_STEPS_X_EP = 10
    NUM_EPISODES = number_of_observation // MAX_STEPS_X_EP
    score = agent.test(env, nb_episodes=NUM_EPISODES, visualize=visualize, nb_max_episode_steps=MAX_STEPS_X_EP, verbose=1)
    return score

def train_agent(env, agent, callbacks):
    TOT_NUM_STEP = 50000  # overall number of steps in the all training
    MAX_STEPS_X_EP = 10  # reset episode when `isdone` or after 10 steps
    return agent.fit(env, nb_steps=TOT_NUM_STEP, nb_max_episode_steps=MAX_STEPS_X_EP, visualize=False, verbose=2, callbacks=callbacks)

def main():
    NUM_OBS_TESTING = 1000
    env = gym.make(ENV_NAME)
    agent, callbacks = build_agent(env.observation_space.shape, env.action_space.n)
    train_score = train_agent(env, agent, callbacks)

    print("TESTING OVER {} TRACES".format(NUM_OBS_TESTING))
    k_mc = estimate_frequency_via_mc(NUM_OBS_TESTING)
    test_score = estimate_frequency_via_agent(NUM_OBS_TESTING, agent)
    k_ag = np.where(np.array(test_score.history['episode_reward'])>=10)[0].shape[0]
    print("[Result] Frequency MC: {} / {} = {} | Frequency RL: {} / {} = {}".format(k_mc, NUM_OBS_TESTING, k_mc/NUM_OBS_TESTING,
                                                                                    k_ag, NUM_OBS_TESTING, k_ag/NUM_OBS_TESTING))
    import ipdb
    ipdb.set_trace()

def test_mc():
    NUM_OBS = 10000
    env = gym.make(ENV_NAME)
    k = 0
    for i in range(NUM_OBS):
        env.reset()
        if env.is_done:
            k = k + 1
    print("Frequency MC: {} / {} = {}".format(k, NUM_OBS, k/NUM_OBS))

if __name__=='__main__':
    #test_environment()
    #test_agent()
    main()
    #test_mc()
