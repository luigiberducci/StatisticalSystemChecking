import os
import random as rand
from time import strftime, gmtime

import numpy as np
import matplotlib.pyplot as plt


def get_future_minimum_array(array):
    """
    Given array A, it returns array B s.t. B[i] = min{A[j] | j>=i}.
    Note: this function has been introduced for computing reward array instead of
    using a single reward value for all the states.
    :param array:   input array
    :return:        array
    """
    out = np.ones(array.shape) * array[-1]
    for i in range(array.shape[0]-2, -1, -1):
        out[i] = min(out[i + 1], array[i])
    return out

class SRSystem:
    """
    Discrete-state model (FSA) with discrete state and time.
    At each time, go to the next state with probability P.
    """

    def __init__(self, p=0.5, init_state=0, goal_state=10, time_horizon=10):
        self.P = p  # transition prob P(x+1|x)
        self.INIT_STATE = init_state
        self.GOAL_STATE = goal_state
        self.TIME_HORIZON = time_horizon
        # Set init state
        self.original_s0 = np.vstack([self.INIT_STATE, 0.0])  # time
        self.x = self.INIT_STATE  # state
        self.t = 0.0  # time
        # History
        self.hx_x = self.x  # state history
        self.hx_t = self.t  # time history
        # Trace information
        self.last_init_state = 0  # init state identified by index in trajectory
        self.i_max_reward = 0  # index of state with max reward
        self.robustness = np.Inf  # robustness value
        self.rob_avg = 1 / 2  # mean rob for Robustness Scaling, init 1/2 in order to avoid division by 0
        self.reward_array = None  # array of reward values for each state in the trace
        self.reward = 0
        self.error_time = None

    def print_config(self):
        print("[Info] Environment: Success Runs")
        print("[Info] Parameters: P(x'|x)={}, Time Horizon {}, Goal State {}".format(self.P,
                                                                                     self.TIME_HORIZON,
                                                                                     self.GOAL_STATE))
        print()

    def set_rob_scaling(self, rob_avg):
        """
        Set the mean min robustness value for Robustness Scaling.
        :param rob_avg:     estimated min robustness
        """
        self.rob_avg = rob_avg

    def set_state(self, x, time):
        # DEPRECATED
        self.x = x
        self.t = time

    def set_prefix(self, prefix):
        # Set state
        self.x = prefix[0, -1]
        self.t = prefix[1, -1]
        # Set history
        self.hx_x = prefix[0, :]
        self.hx_t = prefix[1, :]
        # Rendering
        self.last_init_state = prefix.shape[1] - 1
        self.i_max_reward = prefix.shape[1] - 1
        self.robustness = np.Inf  # robustness value
        self.reward = 0
        self.error_time = None

    def reset_init_state(self):
        # state
        self.x = self.INIT_STATE
        self.t = 0
        # history
        self.hx_x = self.x
        self.hx_t = self.t
        # rendering
        self.error_time = None
        self.last_init_state = 0

    def get_state(self):
        return np.vstack([self.x, self.t])

    def get_trace(self):
        return np.vstack([self.hx_x, self.hx_t])

    def is_current_trace_false(self):
        return self.error_time is not None

    def get_reward(self):
        return self.compute_mc_reward_array(self.GOAL_STATE, self.TIME_HORIZON)

    def terminal_exp_reward(self, X_GOAL, TIME_HORIZON):
        # DEPRECATED
        """
        TODO report calculation
        :param X_GOAL:
        :param TIME_HORIZON:
        :return:
        """
        zeroepsilon = 0.001  # because the invariant is that x_goal-x>0 -> x_goal-x>=`zeroepsilon`
        time_diff = TIME_HORIZON - self.hx_t - zeroepsilon
        state_diff = X_GOAL - self.hx_x - zeroepsilon
        max_array = np.max([time_diff, state_diff], axis=0)
        i_min_rob = np.argmin(max_array[self.last_init_state:])  # best rob from init state
        best_rob_trace = max_array[self.last_init_state + i_min_rob]

        # Flag for falsification condition
        error_mask = (self.hx_x == X_GOAL) & (self.hx_t == TIME_HORIZON)
        error_found = any(error_mask)
        if error_found:
            self.error_time = np.where(error_mask)[0][0]
        self.reward = np.exp(-best_rob_trace)
        # safety check on reward computation
        assert self.reward <= 1 or error_found
        assert not error_found or self.reward > 1
        # update reward information
        self.i_max_reward = self.last_init_state + i_min_rob  # for rendering
        self.robustness = best_rob_trace
        return self.reward, self.error_time

    def compute_mc_reward_array(self, X_GOAL, TIME_HORIZON):
        """
        Compute the `reward_array` R of the current trace, defined inductively as follows:
            R_t = exp(-rho_t)
            R_i = exp(-min{ rho_j | j>=i })     # future min
        """
        zeroepsilon = 0.001  # because the invariant is that x_goal-x>0 -> x_goal-x>=`zeroepsilon`
        time_diff = TIME_HORIZON - self.hx_t - zeroepsilon
        state_diff = X_GOAL - self.hx_x - zeroepsilon
        max_array = np.max([time_diff, state_diff], axis=0)
        self.i_max_reward = np.argmin(max_array)
        self.robustness = max_array[self.i_max_reward]
        # Novelty here: compute future min robustness for each state in the trace
        future_rob_array = get_future_minimum_array(max_array)
        # Robustness scaling
        norm_rob_array = future_rob_array / (2 * self.rob_avg)
        reward_array = np.exp(-norm_rob_array)
        # Error detection
        error_mask = (self.hx_x == X_GOAL) & (self.hx_t == TIME_HORIZON)
        error_found = any(error_mask)
        if error_found:
            self.error_time = np.where(error_mask)[0][0]
        self.reward = reward_array[self.i_max_reward]   # the max reward will be where the robustness is minimum
        self.reward_array = reward_array
        # safety check on reward computation
        assert self.reward <= 1 or error_found
        assert not error_found or self.reward > 1

    def get_min_robustness(self):
        """
        Return the minimal robustness of the current trace as pair `index`, `value`.
        Note: this method has been introduced for (epsilon, delta)-estimation of the mean minimal robustness, in order
        to improve the reward with proper "robustness scaling".
        :return:    `i_min_robustness`: index with min robustness (when it occurs)
                    `robustness`:       value of min robustness
        """
        TIME_HORIZON = self.TIME_HORIZON
        X_GOAL = self.GOAL_STATE
        zeroepsilon = 0.001  # because the invariant is that x_goal-x>0 -> x_goal-x>=`zeroepsilon`
        time_diff = TIME_HORIZON - self.hx_t - zeroepsilon
        state_diff = X_GOAL - self.hx_x - zeroepsilon
        max_array = np.max([time_diff, state_diff], axis=0)
        i_min_robustness = np.argmin(max_array)
        robustness = max_array[self.i_max_reward]
        return i_min_robustness, robustness

    def run_system(self):
        while self.TIME_HORIZON > self.t:
            self.step_system()
            # store trace
            self.hx_x = np.hstack((self.hx_x, self.x))
            self.hx_t = np.hstack((self.hx_t, self.t))
        self.get_reward()

    def step_system(self):
        self.t = self.t + 1
        if self.x == self.GOAL_STATE:
            return
        elif rand.random() <= self.P:
            self.x = self.x + 1

    def render(self, title="", save_fig=False, out_dir="out", prefix="falsification"):
        """
        Render method to show the current state.
        It plots the trajectory of the system: the true trajectory and the belief trajectory.
        :return: -
        """
        state_radius = 0.2
        plt.figure(1)
        ax = plt.gca()
        ax.cla()
        ax.set_title(title)
        # plot state space
        for t in range(self.TIME_HORIZON+1):
            for x in range(0, t+1):
                ax.add_artist(plt.Circle((t, x), color='gray', fill=True, radius=state_radius))
        # plot trajectory
        ax.plot(self.hx_t, self.hx_x, color="orange")  # plot the true trajectory
        for i, (t, x) in enumerate(zip(self.hx_t, self.hx_x)):
            color = 'yellow' if i < self.last_init_state else 'orange'
            final_color = 'red' if self.error_time is None else 'green'
            if self.error_time is not None and i >= self.error_time:
                ax.add_artist(plt.Circle((t, x), color=final_color, fill=True, radius=state_radius))
            elif i == len(self.hx_t)-1:
                ax.add_artist(plt.Circle((t, x), color=final_color, fill=True, radius=state_radius))
            else:
                ax.add_artist(plt.Circle((t, x), color=color, fill=True, radius=state_radius))
        ax.axis("equal")
        ax.grid(True)
        plt.pause(0.001)
        if save_fig:  # save only last plot
            fig_name = "{}_{}".format(prefix, strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
            plt.savefig(os.path.join(out_dir, fig_name))
