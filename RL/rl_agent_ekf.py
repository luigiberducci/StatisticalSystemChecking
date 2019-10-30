import numpy as np
import gym
import gym_ekf_localization
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

ENV_NAME = 'ekf-loc-v0'

def build_model(observation_space_shape, num_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + observation_space_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    print(model.summary())
    return model

def build_agent(obs_space_shape, num_actions):
    model = build_model(obs_space_shape, num_actions)
    replay = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.5)
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=replay, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn

def test_agent(agent=None):
    print("[TEST] Run UNTRAINED AGENT in 1 episode until reach a success (error detected).")
    input("Continue?")
    env = gym.make(ENV_NAME)
    num_actions = env.action_space.n
    obs_space_shape = env.observation_space.shape
    if agent is None:
        print("[Info] Building new agent")
        agent = build_agent(obs_space_shape, num_actions)
    agent.test(env, nb_episodes=1, visualize=True)
    env.holdon_plot()

def test_environment():
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
    #TODO
    return
    env = gym.make(ENV_NAME)
    counter = 0
    for i in range(number_of_observation):
        observation = env.reset()
        if env.isdone:
            counter = counter + 1
    return counter

def estimate_frequency_via_agent(number_of_observation, agent):
    #TODO
    return
    env = gym.make(ENV_NAME)
    counter = 0
    MAX_STEPS_X_EP = 10
    NUM_EPISODES = number_of_observation // MAX_STEPS_X_EP
    while NUM_EPISODES>0:
        agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=MAX_STEPS_X_EP, verbose=2)
        if env.isdone:
            counter = counter + 1
        NUM_EPISODES = NUM_EPISODES - 1
    return counter

def main():
    TOT_NUM_STEP = 50 #overall number of steps in the all training
    MAX_STEPS_X_EP = 10 #reset episode when `isdone` or after 10 steps
    env = gym.make(ENV_NAME)
    agent = build_agent(env.observation_space.shape, env.action_space.n)
    agent.fit(env, nb_steps=TOT_NUM_STEP, nb_max_episode_steps=MAX_STEPS_X_EP, visualize=True, verbose=2)
    print("TESTING")
    test_agent(agent)
    import ipdb
    ipdb.set_trace()

if __name__=='__main__':
    #test_environment()
    #test_agent()
    main()
