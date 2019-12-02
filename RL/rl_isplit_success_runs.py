import numpy as np
import os
import gym
import gym_success_runs

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, SimpleRNN
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

from ImportanceSplittingCallback import ImportanceSplittingCallback

import ipdb
import argparse

tf.compat.v1.disable_eager_execution()

#def nonzero_init_weights():
#   return initializers.RandomNormal(mean=-1.0, stddev=0.05, seed=None)

GLOBAL_ENV_NAME='succruns-v1'

def build_model(observation_space_shape, num_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + observation_space_shape))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    print(model.summary())
    return model

def build_agent(observation_space_shape, num_actions):
    # Experience replay
    WARMUP_STEPS = 500     # Collect the first steps before start experience replay
    MEM_LIMIT = 5000        # Max number of steps to store
    MEM_WINDOW_LEN = 1      # Experience of lenght 1 (single step)
    # Target network
    TARGET_MODEL_UPD_RATE = 1e-2    # Update target network with this rate
    # Build network, exp. replay and policy
    #model = build_model(observation_space_shape, num_actions)
    model = build_model(observation_space_shape, num_actions)
    replay = SequentialMemory(limit=MEM_LIMIT, window_length=MEM_WINDOW_LEN)
    policy = GreedyQPolicy()

    # Finally build the agent
    GAMMA = 1
    dqn = DQNAgent(model=model, gamma=GAMMA, nb_actions=num_actions, memory=replay, nb_steps_warmup=WARMUP_STEPS,
                   target_model_update=TARGET_MODEL_UPD_RATE, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn

def build_importance_splitting(env, agent, run_ispl=True, outdir='out/succruns'):
    return ImportanceSplittingCallback(env, agent, run_ispl=run_ispl, outdir=outdir)

def get_model_checkpoint_callback(outdir, interval):
    model_dir = os.path.join(outdir, "models")
    if not os.path.exists(model_dir):  # Create if necessary
        os.makedirs(model_dir, exist_ok=True)
    fp = os.path.join(model_dir, 'weights.{step:02d}.hdf5')
    return ModelIntervalCheckpoint(filepath=fp, interval=interval, verbose=1)

def run(ENV_NAME='succruns-v1', NUM_EPISODES=1000, NUM_STEP_X_EPISODE=10, run_ispl=True, outdir='out', save_models=False, save_interval=None):
    search_algo = "Importance Splitting" if run_ispl else "Uniform Random Simulation"
    # Create Environment and Agent
    env = gym.make(ENV_NAME, P=0.5)
    agent = build_agent(env.observation_space.shape, env.action_space.n)
    # Create callbacks
    imp_spl = build_importance_splitting(env, agent, run_ispl=run_ispl, outdir=outdir)
    callbacks_list = [imp_spl]
    if save_models:
        save_callback = get_model_checkpoint_callback(outdir, interval=save_interval)
        callbacks_list.append(save_callback)
    # Start training
    print("[Info] Start {} on {}.".format(search_algo, ENV_NAME))
    agent.fit(env, nb_steps=NUM_EPISODES*NUM_STEP_X_EPISODE, nb_max_episode_steps=None,
              callbacks=callbacks_list, verbose=2, visualize=False)
    print("[Info] End {}. Falsification occurred {} times.".format(search_algo, imp_spl.falsification_counter))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', default='ISplit', help='ISplit or MC')
    parser.add_argument('--episodes', default=1000, nargs='?', type=int, help='Number of episodes (default 1000)')
    parser.add_argument('--step', default=10, nargs='?', type=int, help='Number of step per episode (default 10 sec=100 steps)')
    parser.add_argument('--outdir', default='out', nargs='?', help='Output directory for logging')
    parser.add_argument('--interval', default=1000, nargs='?', type=int, help='Checkpoint interval (# steps) in which save model')

    args = parser.parse_args()
    run_ispl = True
    save_models = False
    save_interval = None
    if args.search == 'MC':
        run_ispl = False
    if args.interval:
        save_interval = args.interval
        save_models = True
    run(ENV_NAME=GLOBAL_ENV_NAME, NUM_EPISODES=args.episodes, NUM_STEP_X_EPISODE=args.step,
        run_ispl=run_ispl, outdir=args.outdir, save_models=save_models, save_interval=save_interval)
    return(0)

if __name__=="__main__":
    main()
