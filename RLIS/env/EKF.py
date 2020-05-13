import os
from time import strftime, gmtime

import numpy as np
import math
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

class EKFSystem:
    def __init__(self, time_horizon=10.0, transient_time=3.0, err_threshold=0.7):
        # Covariance for EKF simulation
        self.Q = np.diag([
            0.1,  # variance of location on x-axis
            0.1,  # variance of location on y-axis
            np.deg2rad(1.0),  # variance of yaw angle
            1.0  # variance of velocity
        ]) ** 2  # predict state covariance
        self.R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

        #  Simulation parameter
        self.INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
        self.GPS_NOISE = np.diag([0.5, 0.5]) ** 2
        self.DT = 0.1  # time tick [s]
        self.time_event = 5.0  # at this time, the system change orientation
        # Safety property parameters
        self.TIME_HORIZON = time_horizon   # simulation time [s]
        self.ERR_THRESHOLD = err_threshold  #threshold for error detection
        self.TRANSIENT_TIME = transient_time  #initial unstable period

        # State variables
        self.x_true = np.zeros((4, 1))
        self.x_est = np.zeros((4, 1))
        self.p_est = np.eye(4)
        self.time = 0.0
        self.original_s0 = np.vstack([np.zeros((4, 1)),  # x_true
                                      np.zeros((4, 1)),  # x_est
                                      np.eye(4).reshape((16, 1)),  # p_est
                                      0.0])  # time
        self.sensors_data = np.zeros((2, 1))  # data from gps
        # History
        self.hx_true = self.x_true
        self.hx_est = self.x_est
        self.hx_time = self.time
        self.hx_p_est = self.p_est.reshape((16, 1))  # flat p_est matrix
        # Rendering
        self.last_init_state = 0    # init state identified by index in trajectory
        self.i_max_reward = 0       # index of state with max reward
        self.robustness = np.Inf    # robustness value
        self.rob_avg = 1 / 2  # init 1/2 is to compensate division
        self.reward_array = None  # array of reward values for each state in the trace
        self.reward = 0
        self.error_time = None

    def print_config(self):
        print("[Info] Environment: EKF Localization")
        print("[Info] Parameters: Error Threshold {}, Time Horizon {}, Transient Time {}".format(self.ERR_THRESHOLD,
                                                                                                 self.TIME_HORIZON,
                                                                                                 self.TRANSIENT_TIME))
        print()

    def set_rob_scaling(self, rob_avg):
        """
        Set the mean min robustness value for Robustness Scaling.
        :param rob_avg:     estimated min robustness
        """
        self.rob_avg = rob_avg

    def set_state(self, x_true, x_est, p_est, time):
        # DEPRECATED, WORK ONLY WITH PREFIXES
        self.x_true = x_true
        self.x_est = x_est
        self.p_est = p_est
        self.time = time

    def set_prefix(self, sampled_prefix):
        # Set state
        self.x_true = sampled_prefix[0:4, -1].reshape((4, 1))
        self.x_est = sampled_prefix[4:8, -1].reshape((4, 1))
        self.p_est = sampled_prefix[8:24, -1].reshape((4, 4))
        self.time = sampled_prefix[24:25, -1]
        # Set history
        self.hx_true = sampled_prefix[0:4, :]
        self.hx_est = sampled_prefix[4:8, :]
        self.hx_p_est = sampled_prefix[8:24, :]
        self.hx_time = sampled_prefix[24:25, :][0]  #avoid dimension mismatch
        # Rendering
        self.last_init_state = sampled_prefix.shape[1] - 1
        self.i_max_reward = sampled_prefix.shape[1] - 1
        self.robustness = np.Inf  # robustness value
        self.reward = 0
        self.error_time = None

    def reset_init_state(self):
        # State variables
        self.x_true = np.zeros((4, 1))
        self.x_est = np.zeros((4, 1))
        self.p_est = np.eye(4)
        self.time = 0.0
        # History
        self.hx_true = self.x_true
        self.hx_est = self.x_est
        self.hx_time = self.time
        self.hx_p_est = self.p_est.reshape((16, 1))  # flat p_est matrix
        # Rendering
        self.last_init_state = 0  # init state identified by index in trajectory
        self.i_max_reward = 0     # index of state with max reward
        self.robustness = np.Inf  # robustness value
        self.error_time = None

    def get_state(self):
        """
        Return the current state (exec. trace) as a unique array (stack true, est, time arrays)
        :return: stacked representation of the current state
        """
        return np.vstack([self.x_true, self.x_est, self.p_est.reshape((16, 1)), self.time])

    def get_trace(self):
        """
        Return the trace (trajectory) which leads to the current state.
        :return: trace of state variables
        """
        return np.vstack([self.hx_true, self.hx_est, self.hx_p_est, self.hx_time])

    def is_current_trace_false(self):
        return self.error_time is not None

    def get_reward(self):
        return self.compute_mc_reward_array(self.ERR_THRESHOLD, self.TRANSIENT_TIME, self.TIME_HORIZON)

    def terminal_exp_reward(self, ERR_THRESHOLD, TRANSIENT_TIME, TIME_HORIZON):
        """
        Reward defined on formula phi = G p, where p = m>=0 | -p | p or q
        as reward(x) = 0 if x is not terminal, exp(-rob(trace(x,:))) otherwise
        with rob(trace, phi) = min_{x in trace} rob(x, phi) and rob(x) defined as
            rob(x, m>=0) = eval(m, x)
            rob(x, -p) = -rob(x, p)
            rob(x, p or q) = max(rob(x, p), rob(x, q))
        where eval(m, x) = m with value(x) instead of x
        :param ERR_THRESHOLD:   threshold for error detection
        :param TRANSIENT_TIME: time when finish the transient phase in which the localization is unstable
        :param TIME_HORIZON: total lenght of trace
        :return: `isdone` if the trace is complete (time or succ), `rewar/FALSd`, eventual `error_time` in which error occurred
        """
        trans_minus_time = TRANSIENT_TIME - self.hx_time
        location_difference = np.linalg.norm(self.hx_true[0:2, :] - self.hx_est[0:2, :], axis=0)
        thresh_minus_err = ERR_THRESHOLD - location_difference
        max_array = np.max([trans_minus_time, thresh_minus_err], axis=0)
        i_min_rob = np.argmin(max_array[self.last_init_state:])     # only from the starting point
        best_rob_trace = max_array[self.last_init_state + i_min_rob]  #max because rob should be minimized

        # Flag for falsification condition
        error_mask = (self.hx_time > TRANSIENT_TIME) & (location_difference > ERR_THRESHOLD)
        error_found = any(error_mask)
        if error_found:
            self.error_time = np.where(error_mask)[0][0]
        self.reward = np.exp(-best_rob_trace)
        # safety check on reward computation
        assert self.reward <= 1 or error_found
        assert not error_found or self.reward > 1
        # update reward information
        self.i_max_reward = self.last_init_state + i_min_rob   #for rendering
        self.robustness = best_rob_trace
        return self.reward, self.error_time

    def compute_mc_reward_array(self, ERR_THRESHOLD, TRANSIENT_TIME, TIME_HORIZON):
        """
        Compute the `reward_array` R of the current trace, defined inductively as follows:
            R_t = exp(-rho_t)
            R_i = exp(-min{ rho_j | j>=i })     # future min
        """
        trans_minus_time = TRANSIENT_TIME - self.hx_time
        location_difference = np.linalg.norm(self.hx_true[0:2, :] - self.hx_est[0:2, :], axis=0)
        thresh_minus_err = ERR_THRESHOLD - location_difference
        max_array = np.max([trans_minus_time, thresh_minus_err], axis=0)
        self.i_max_reward = np.argmin(max_array)
        self.robustness = max_array[self.i_max_reward]
        # Novelty here: compute future min robustness for each state in the trace
        future_rob_array = get_future_minimum_array(max_array)
        # Robustness scaling
        norm_rob_array = future_rob_array / (2 * self.rob_avg)
        reward_array = np.exp(-norm_rob_array)
        # Error detection
        error_mask = (self.hx_time > TRANSIENT_TIME) & (location_difference > ERR_THRESHOLD)
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
        ERR_THRESHOLD = self.ERR_THRESHOLD
        TRANSIENT_TIME = self.TRANSIENT_TIME
        trans_minus_time = TRANSIENT_TIME - self.hx_time
        location_difference = np.linalg.norm(self.hx_true[0:2, :] - self.hx_est[0:2, :], axis=0)
        thresh_minus_err = ERR_THRESHOLD - location_difference
        max_array = np.max([trans_minus_time, thresh_minus_err], axis=0)
        i_min_robustness = np.argmin(max_array)
        robustness = max_array[i_min_robustness]
        return i_min_robustness, robustness

    def run_system(self):
        while self.TIME_HORIZON > self.time:
            self.step_system()
            # store data history
            self.hx_true = np.hstack((self.hx_true, self.x_true))
            self.hx_est = np.hstack((self.hx_est, self.x_est))
            self.hx_p_est = np.hstack((self.hx_p_est, self.p_est.reshape(16, 1)))  # not used now
            self.hx_time = np.hstack((self.hx_time, self.time))    #all this to avoid dimension mistmatch
        self.get_reward()

    def step_system(self):
        self.time += self.DT
        u = self.my_calc_input(self.time)
        x_dr = self.x_true      # to maintain previous method
        self.x_true, z, xDR, ud = self.observation(self.x_true, x_dr, u)
        self.sensors_data = np.hstack((self.sensors_data, z))
        self.x_est, self.p_est = self.ekf_estimation(self.x_est, self.p_est, z, ud)

    def ekf_estimation(self, xEst, PEst, z, u):
        #  Predict
        xPred = self.motion_model(xEst, u)
        jF = self.jacob_f(xPred, u)
        PPred = jF @ PEst @ jF.T + self.Q

        #  Update
        jH = self.jacob_h()
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + self.R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst, PEst

    def observation(self, xTrue, xd, u):
        xTrue = self.motion_model(xTrue, u)

        # add noise to gps x-y
        z = self.observation_model(xTrue) + self.GPS_NOISE @ np.random.randn(2, 1)

        # add noise to input
        ud = u + self.INPUT_NOISE @ np.random.randn(2, 1)

        xd = self.motion_model(xd, ud)

        return xTrue, z, xd, ud

    def motion_model(self, x, u):
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[self.DT * math.cos(x[2, 0]), 0],
                      [self.DT * math.sin(x[2, 0]), 0],
                      [0.0, self.DT],
                      [1.0, 0.0]])

        x = F @ x + B @ u

        return x

    def observation_model(self, x):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = H @ x

        return z

    def jacob_f(self, x, u):
        """
        Jacobian of Motion Model

        motion model
        x_{t+1} = x_t+v*dt*cos(yaw)
        y_{t+1} = y_t+v*dt*sin(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = -v*dt*sin(yaw)
        dx/dv = dt*cos(yaw)
        dy/dyaw = v*dt*cos(yaw)
        dy/dv = dt*sin(yaw)
        """
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, -self.DT * v * math.sin(yaw), self.DT * math.cos(yaw)],
            [0.0, 1.0,  self.DT * v * math.cos(yaw), self.DT * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        return jF

    def jacob_h(self):
        # Jacobian of Observation Model
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return jH

    def my_calc_input(self, time):
        v = 1.5  # [m/s]
        yawrate = 0.35  # [rad/s]
        if time >= self.time_event:
            yawrate = -0.35  # [rad/s]
        u = np.array([[v], [yawrate]])
        return u

    def render(self, title="", save_fig=False, out_dir="out", prefix="falsification"):
        """
        Render method to show the current state.
        It plots the trajectory of the system: the true trajectory and the belief trajectory.
        :return: -
        """
        plt.figure(1)
        ax = plt.gca()
        ax.cla()
        ax.set_title(title)
        ax.plot(self.hx_true[0, :].flatten(),
                self.hx_true[1, :].flatten(), "-g", label="True state")  # plot the true trajectory
        ax.plot(self.hx_est[0, :].flatten(),
                self.hx_est[1, :].flatten(), "-b", label="Belief state")  # plot the estimated (belief) trajectory
        ax.legend()
        ax.scatter(self.hx_true[0, self.last_init_state].flatten(),
                   self.hx_true[1, self.last_init_state].flatten(), marker='o')  # mark last break-point (action)
        ax.scatter(self.hx_est[0, self.last_init_state].flatten(),
                   self.hx_est[1, self.last_init_state].flatten(), marker='o')
        if self.error_time is not None:
            true_value_coord = (self.hx_true[0, self.error_time].flatten(), self.hx_true[1, self.error_time].flatten())
            ax.scatter(self.hx_true[0, self.error_time].flatten(),
                       self.hx_true[1, self.error_time].flatten(), marker='X')  # mark the first error found
            ax.scatter(self.hx_est[0, self.error_time].flatten(),
                       self.hx_est[1, self.error_time].flatten(), marker='X')
            ax.add_artist(plt.Circle(true_value_coord, color='r', fill=False, radius=self.ERR_THRESHOLD))
        else:
            # plot max reward
            true_value_coord = (self.hx_true[0, self.i_max_reward].flatten(), self.hx_true[1, self.i_max_reward].flatten())
            ax.scatter(self.hx_true[0, self.i_max_reward].flatten(),
                       self.hx_true[1, self.i_max_reward].flatten(), marker='+')  # mark the first error found
            ax.scatter(self.hx_est[0, self.i_max_reward].flatten(),
                       self.hx_est[1, self.i_max_reward].flatten(), marker='+')

        ax.axis("equal")
        ax.grid(True)
        plt.pause(0.001)
        if save_fig:  # save only last plot
            fig_name = "{}_{}".format(prefix, strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
            plt.savefig(os.path.join(out_dir, fig_name))
