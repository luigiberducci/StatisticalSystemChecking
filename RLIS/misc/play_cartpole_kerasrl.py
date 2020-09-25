from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents import DQNAgent
import gym

import tensorflow as tf

print(tf.__version__)
def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model

env = gym.make("CartPole-v1")
num_actions = env.action_space.n
model = build_model(env.observation_space.shape[0], num_actions)
model.compile(Adam(lr=1e-3), metrics=['mae'])

memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
                              value_test=.05, nb_steps=10000)

dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
               enable_double_dqn=False, policy=policy)

dqn.fit(env, nb_steps=500, visualize=False, verbose=2)