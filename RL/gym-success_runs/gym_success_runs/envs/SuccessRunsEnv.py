import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import math
import random as rand

class SuccessRunsEnv(gym.Env):
    def __init__(self, P=0.5):
        self.sys = FSA(P)
        DUMMY_ACTION = 1
        OBS_VARS = 2    #state and time

        self.action_space = spaces.Discrete(DUMMY_ACTION)
        self.observation_space = spaces.Box(low=0,
                                            high=np.Inf,
                                            shape=(OBS_VARS, DUMMY_ACTION), dtype=np.float16)
        self.is_done = False
        self.reward = 0
        # Set of initial prefixes (IMPORTANCE SPLITTING)
        self.prefix_list = list()
        self.original_s0 = np.vstack([0, 0])  # state, time
        self.prefix_list.append(self.original_s0)
        self.best_rob_trace = np.Inf
        # Init history for rendering
        self.last_init_state = 0  # init state identified by index in trajectory
        self.error_time = None  # index of first state in which error occurred
        self.hx_x = self.sys.x
        self.hx_time = self.sys.t

    def print_info_config(self):
        print("[Info] Environment: Success Runs")
        print("[Info] Parameters: P(x'|x)={}".format(self.sys.P))

    def get_state(self):
        """
        Return the current state as a unique array (stack true, est, time arrays)
        :return: stacked representation of the current state
        """
        return np.vstack([self.sys.x, self.sys.t])

    def get_trace(self):
        """
        Return the trace (trajectory) which leads to the current state.
        :return: trace of state variables
        """
        return np.vstack([self.hx_x, self.hx_time])

    def is_current_trace_false(self):
        """
        Return a bool value which indicates if the trace contains error or not.
        :return: boolean monitor (true: error found, false: error not found)
        """
        return self.error_time is not None

    def step(self, useless_action):
        """
        :param action_prefix: useless dummy action
        :return: state, reward, id_done flag, and info
        """
        self.sys.step_system()  # Step run ahed
        self.is_done, self.reward, self.error_time = self.get_reward()
        # store data history for rendering
        self.hx_x = np.hstack((self.hx_x, self.sys.x))
        self.hx_time = np.hstack((self.hx_time, self.sys.t))  # all this to avoid dimension mistmatch
        return self.get_state(), self.reward, self.is_done, {}

    def sample_init_state(self):
        # `prefix_list`  is a list of prefixes, sample one of them uniformly at random
        sampled_prefix = rand.sample(self.prefix_list, 1)[0]  # sample return a list of 1 (k) element
        last_state = sampled_prefix.shape[1] - 1  # index of last state of prefix
        # Set state
        x = sampled_prefix[0, last_state]
        time = sampled_prefix[1, last_state]
        # Set history
        self.hx_x = sampled_prefix[0, :]
        self.hx_time = sampled_prefix[1, :]
        self.error_time = None
        self.last_init_state = last_state
        # Copy is necessary otherwise the object in memory is the same and happened unexpected changes
        return x.copy(), time.copy()

    def add_prefix(self, prefix):
        self.prefix_list.append(prefix)

    def clear_prefix_list(self):
        self.prefix_list.clear()

    def reset_original_prefix_list(self):
        self.prefix_list.clear()
        self.prefix_list.append(self.original_s0)

    def reset(self):
        """
        Reset the state of the system.
        :return: initial state
        """
        x, time = self.sample_init_state()
        self.sys.set_state(x, time)
        self.is_done = False
        self.best_rob_trace = np.Inf
        return self.get_state()

    def render(self, mode='human'):
        """
        Render method to show the current state.
        It plots the trajectory of the system: the true trajectory and the belief trajectory.
        :return: -
        """
        ax = plt.gca()
        ax.cla()
        ax.plot(self.hx_x, "-g")  # plot the true trajectory
        ax.scatter(self.last_init_state, self.hx_x[self.last_init_state], marker='o')  # mark last break-point (action)
        if self.error_time is not None:
            ax.scatter(self.error_time, self.hx_x[self.error_time], marker='X')  # mark the first error found
        ax.axis("equal")
        ax.grid(True)
        plt.pause(0.001)

    def get_reward(self):
        X_GOAL = 10     #goal state
        TIME_HORIZON = 10.0  # overall time horizon
        return self.compute_exp_reward(X_GOAL, TIME_HORIZON)

    def compute_exp_reward(self, X_GOAL, TIME_HORIZON):
        """
        Reward defined on formula p = m>=0 | -p | p or q
        as reward(phi) = exp(-rob(phi)) - 1, where rob is
            rob(m>=0) = exp(-eval(m)) - 1
            rob(-p) = -rob(p)
            rob(p or q) = max(rob(p), rob(q))
        where eval(m) = m
        :return: `isdone` if the trace is complete (time or succ), `reward`, eventual `error_time` in which error occurred
        """
        zeroepsilon = 0.001     # because the invariant is that x_goal-x>0 -> x_goal-x>=`zeroepsilon`
        time_diff = TIME_HORIZON - self.sys.t
        state_diff = self.sys.GOAL_STATE - self.sys.x - zeroepsilon
        self.best_rob_trace = np.min([self.best_rob_trace, np.max([time_diff, state_diff])])  #max because rob should be minimized

        # Flag for falsification condition
        error_found = (self.sys.x == self.sys.GOAL_STATE) & (self.sys.t==TIME_HORIZON)
        time_elapsed = True if self.sys.t >= TIME_HORIZON else False
        error_time = self.error_time
        if error_time is None:
            error_time = None if not error_found else self.hx_x.shape[0]  # Notice: not shape-1 because the errortime is the current (not yet appended)
        # is_done = error_found or time_elapsed
        is_done = time_elapsed  #the end of a episode is always when time elapsed

        reward = 0
        if is_done:
            reward = np.exp(-self.best_rob_trace)
        return is_done, reward, error_time

class FSA:
    """
    Dummy model of FSA with discrete state and time.
    At each time, go to the next state with probability P, and to state 0 with probability 1-P
    """
    def __init__(self, P=0.5):
        self.INIT_STATE = 0
        self.GOAL_STATE = 10
        self.SINK_STATE = -1
        self.TIME_TO_REACH_GOAL = 10
        self.x = self.INIT_STATE  #state
        self.t = 0  #time
        self.P = P  #transition prob P(x+1|x)

    def reset_init_state(self):
        self.x = self.INIT_STATE
        self.t = 0

    def set_state(self, x, time):
        self.x = x
        self.t = time

    def step_system(self):
        self.t = self.t + 1
        if self.x in [self.GOAL_STATE, self.SINK_STATE]:
            return
        #if self.t >= self.TIME_TO_REACH_GOAL and (self.x + 1 < self.GOAL_STATE):
        #    self.x = self.SINK_STATE
        elif rand.random() <= self.P:
            self.x = self.x + 1

    def run_system(self, num_steps=10):
        for i in range(num_steps):
            self.step_system()


