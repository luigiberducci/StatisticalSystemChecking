import os
from time import strftime, gmtime

import numpy as np
from numba import jit, jitclass
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
    for i in range(array.shape[0] - 2, -1, -1):
        out[i] = min(out[i + 1], array[i])
    return out


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    z = H @ x
    return z


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return jH


class TRSystem:
    """
    Trajectory Rollout (TR) system. Online planning based on current pose estimation with ekf.
    """

    def __init__(self, time_horizon=10.0):
        # Covariance for EKF simulation
        self.Q = np.diag([
            0.1,  # variance of location on x-axis
            0.1,  # variance of location on y-axis
            np.deg2rad(1.0),  # variance of yaw angle
            1.0  # variance of velocity
        ]) ** 2  # predict state covariance
        self.R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

        #  Simulation parameter
        self.INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 10
        self.GPS_NOISE = np.diag([0.5, 0.5]) ** 4
        self.TIME_HORIZON = time_horizon  # simulation time [s]
        self.DT = 0.1  # time tick [s]
        self.v = 1.5  # set it to 0 when reach goal
        self.apply_interval = 2  # number of step of application of the last input computed
        self.obstacle_coord = [[2.0, +0.0],  # obstacles xy-coordinates
                               [2.5, -0.5],
                               [3.0, -1.0],
                               [6.0, -2.5],
                               [6.5, -2.0],
                               [7.0, -1.5]]
        self.x_goal = np.array([9, -2])  # goal xy-coordinates
        self.obstacle_radius = 0.25  # radius of obstacles, car and goal (for collision detection)
        self.car_radius = 0.1
        self.goal_radius = 0.25
        self.safe_margin = 0.15

        # State variables
        self.x_true = np.zeros((4, 1))
        self.x_est = np.zeros((4, 1))
        self.p_est = np.eye(4)
        self.time = 0.0
        self.last_u = np.array([[self.v], [0]])
        self.original_s0 = np.vstack([np.zeros((4, 1)),  # x_true
                                      np.zeros((4, 1)),  # x_est
                                      np.eye(4).reshape((16, 1)),  # p_est
                                      0.0,  # time
                                      np.array([[self.v], [0]])])  # input u
        # History
        self.hx_true = self.x_true
        self.hx_est = self.x_est
        self.hx_time = self.time
        self.hx_p_est = self.p_est.reshape((16, 1))  # flat p_est matrix
        self.hx_u = self.last_u
        # Rendering
        self.last_init_state = 0  # init state identified by index in trajectory
        self.i_max_reward = 0  # index of state with max reward
        self.robustness = np.Inf  # best (min) robustness value
        self.rob_avg = 1 / 2  # mean rob for Robustness Scaling, init 1/2 in order to avoid division by 0
        self.reward_array = None  # array of reward values for each state in the trace
        self.reward = 0  # best (max) reward obtained in the current trace
        self.error_time = None

    def print_config(self):
        print("[Info] Environment: Trajectory Rollout (TR)")
        print("[Info] Parameters: Time Horizon {}".format(self.TIME_HORIZON))
        print()

    def set_rob_scaling(self, rob_avg):
        """
        Set the mean min robustness value for Robustness Scaling.
        :param rob_avg:     estimated min robustness
        """
        self.rob_avg = rob_avg

    def set_state(self, x_true, x_est, p_est, time, last_u):
        # DEPRECATED, WORK ONLY WITH PREFIXES
        self.x_true = x_true
        self.x_est = x_est
        self.p_est = p_est
        self.time = time
        self.last_u = last_u

    def set_prefix(self, sampled_prefix):
        # Set state
        self.x_true = sampled_prefix[0:4, -1].reshape((4, 1))
        self.x_est = sampled_prefix[4:8, -1].reshape((4, 1))
        self.p_est = sampled_prefix[8:24, -1].reshape((4, 4))
        self.time = sampled_prefix[24:25, -1]
        self.last_u = sampled_prefix[25:27, -1].reshape(2, 1)
        # Set history
        self.hx_true = sampled_prefix[0:4, :]
        self.hx_est = sampled_prefix[4:8, :]
        self.hx_p_est = sampled_prefix[8:24, :]
        self.hx_time = sampled_prefix[24:25, :][0]  # avoid dimension mismatch
        self.hx_u = sampled_prefix[25:27, :]
        # Rendering
        self.last_init_state = sampled_prefix.shape[1] - 1
        self.i_max_reward = sampled_prefix.shape[1] - 1
        self.robustness = np.Inf  # robustness value
        self.reward = 0
        self.reward_array = None
        self.error_time = None

    def reset_init_state(self):
        # State variables
        self.x_true = np.zeros((4, 1))
        self.x_est = np.zeros((4, 1))
        self.p_est = np.eye(4)
        self.time = 0.0
        self.last_u = np.array([[self.v], [0]])
        # History
        self.hx_true = self.x_true
        self.hx_est = self.x_est
        self.hx_time = self.time
        self.hx_p_est = self.p_est.reshape((16, 1))  # flat p_est matrix
        self.hx_u = self.last_u
        # Rendering
        self.last_init_state = 0  # init state identified by index in trajectory
        self.error_time = None

    def get_state(self):
        """
        Return the current state (exec. trace) as a unique array (stack true, est, time arrays)
        :return: stacked representation of the current state
        """
        return np.vstack([self.x_true, self.x_est, self.p_est.reshape((16, 1)), self.time, self.last_u])

    def get_trace(self):
        """
        Return the trace (trajectory) which leads to the current state.
        :return: trace of state variables
        """
        return np.vstack([self.hx_true, self.hx_est, self.hx_p_est, self.hx_time, self.hx_u])

    def is_current_trace_false(self):
        return self.error_time is not None

    def get_reward(self):
        self.compute_mc_reward_array()

    def terminal_exp_reward(self):
        # DEPRECATED
        """
        Exponential reward e^{-rob}, where rob is the best (min) robustness in the current trace considering only the
        states from the current starting point.
        :return:    trace reward,
                    error time or None (if error isn't occurred)
        """
        collision_range = (self.obstacle_radius + self.car_radius)  # obstacle size + car size
        dist = []  # collect distances from each obstacle, for each state in the trace
        for coord in self.obstacle_coord:
            dist.append(np.linalg.norm(self.hx_true[0:2, :] - np.array(coord).reshape(2, 1), axis=0))
        dist = np.array(dist)
        max_array = - np.max(collision_range - dist, axis=0)
        i_min_rob = np.argmin(max_array[self.last_init_state:])  # only from the starting point
        best_rob_trace = max_array[self.last_init_state + i_min_rob]  # max because rob should be minimized

        # Error detection
        dist = np.min(dist, axis=0)
        error_mask = dist < collision_range
        error_found = error_mask.any()
        if error_found:
            self.error_time = np.where(error_mask)[0][0]
        # Robustness scaling
        norm_rob_trace = best_rob_trace / (2 * self.rob_avg)
        self.reward = np.exp(-norm_rob_trace)
        # safety check on reward computation
        assert self.reward <= 1 or error_found
        assert not error_found or self.reward > 1
        # update reward information
        self.i_max_reward = self.last_init_state + i_min_rob  # for rendering
        self.robustness = best_rob_trace
        return self.reward, self.error_time

    def compute_mc_reward_array(self):
        """
        Compute the `reward_array` R of the current trace, defined inductively as follows:
            R_t = exp(-rho_t)
            R_i = exp(-min{ rho_j | j>=i })     # future min
        """
        collision_range = (self.obstacle_radius + self.car_radius)  # obstacle size + car size
        dist = []  # collect distances from each obstacle, for each state in the trace
        for coord in self.obstacle_coord:
            dist.append(np.linalg.norm(self.hx_true[0:2, :] - np.array(coord).reshape(2, 1), axis=0))
        dist = np.array(dist)
        max_array = - np.max(collision_range - dist, axis=0)
        self.i_max_reward = np.argmin(max_array)
        self.robustness = max_array[self.i_max_reward]
        # Novelty here: compute future min robustness for each state in the trace
        future_rob_array = get_future_minimum_array(max_array)
        # Robustness scaling
        norm_rob_array = future_rob_array / (2 * self.rob_avg)
        reward_array = np.exp(-norm_rob_array)
        # Error detection
        dist = np.min(dist, axis=0)
        error_mask = dist < collision_range
        error_found = error_mask.any()
        if error_found:
            self.error_time = np.where(error_mask)[0][0]
        self.reward = reward_array[self.i_max_reward]  # the max reward will be where the robustness is minimum
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
        collision_range = (self.obstacle_radius + self.car_radius)  # obstacle size + car size
        dist = []  # collect distances from each obstacle, for each state in the trace
        for coord in self.obstacle_coord:
            dist.append(np.linalg.norm(self.hx_true[0:2, :] - np.array(coord).reshape(2, 1), axis=0))
        dist = np.array(dist)
        max_array = - np.max(collision_range - dist, axis=0)
        i_min_robustness = np.argmin(max_array)
        robustness = max_array[self.i_max_reward]
        return i_min_robustness, robustness

    def run_system(self):
        while self.TIME_HORIZON > self.time:
            if self.check_collision(self.x_true[0:2].reshape(2)):
                break
            if self.goal_reached(self.x_true[0:2].reshape(2)):
                break
            self.step_system()
            # store data history
            self.hx_true = np.hstack((self.hx_true, self.x_true))
            self.hx_est = np.hstack((self.hx_est, self.x_est))
            self.hx_p_est = np.hstack((self.hx_p_est, self.p_est.reshape(16, 1)))  # not used now
            self.hx_time = np.hstack((self.hx_time, self.time))  # all this to avoid dimension mistmatch
            self.hx_u = np.hstack((self.hx_u, self.last_u))  # all this to avoid dimension mistmatch
        self.get_reward()

    def step_system(self):
        if (self.hx_true.shape[1] - 1) % self.apply_interval == 0:
            u = self.my_calc_input(self.x_est)  # recompute input
            self.last_u = u
        else:
            u = self.last_u  # during apply_interval, use last u computed
        self.time += self.DT
        self.x_true, z, ud = self.observation(self.x_true, u)
        self.x_est, self.p_est = self.ekf_estimation(self.x_est, self.p_est, z, ud)

    def ekf_estimation(self, xEst, PEst, z, u):
        #  Predict
        xPred = self.motion_model(xEst, u)
        jF = self.jacob_f(xPred, u)
        PPred = jF @ PEst @ jF.T + self.Q

        #  Update
        jH = jacob_h()
        zPred = observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + self.R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst, PEst

    def observation(self, xTrue, u):
        xTrue = self.motion_model(xTrue, u)
        # add noise to gps x-y
        z = observation_model(xTrue) + self.GPS_NOISE @ np.random.randn(2, 1)
        # add noise to input
        # ud = u + self.INPUT_NOISE @ np.random.randn(2, 1)
        ud = u
        return xTrue, z, ud

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

    def motion_model_array(self, x_u_array):
        # parallelize motion model, this function is called with apply_along_axis
        x = x_u_array[0:-2].reshape(4, 1)
        u = x_u_array[-2:].reshape(2, 1)
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[self.DT * math.cos(x[2, 0]), 0],
                      [self.DT * math.sin(x[2, 0]), 0],
                      [0.0, self.DT],
                      [1.0, 0.0]])
        x = F @ x + B @ u
        return np.vstack([x, u]).reshape(6)

    def check_collision_array(self, x_u_array):
        x_u_array = np.tile(x_u_array[0:2].reshape(2, 1), len(self.obstacle_coord))
        obs_state = np.array(self.obstacle_coord).T
        dist = np.linalg.norm(obs_state - x_u_array, axis=0)
        return any(dist <= (self.obstacle_radius + self.car_radius))

    def check_safe_collision_array(self, x_u_array):
        x_u_array = np.tile(x_u_array[0:2].reshape(2, 1), len(self.obstacle_coord))
        obs_state = np.array(self.obstacle_coord).T
        dist = np.linalg.norm(obs_state - x_u_array, axis=0)
        return any(dist <= (self.obstacle_radius + self.car_radius + self.safe_margin))

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
            [0.0, 1.0, self.DT * v * math.cos(yaw), self.DT * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        return jF

    def my_calc_input(self, x_est):
        # import ipdb
        # ipdb.set_trace()
        num_input_samples = 5
        prediction_range = 8
        v = self.v  # [m/s]
        best_u = self.last_u
        best_score = np.Inf
        # import ipdb
        # ipdb.set_trace()
        # self.render()
        avail_u = np.linspace(-math.pi / 2, math.pi / 2, num_input_samples)
        for i in range(num_input_samples):
            u = np.array([[v], [avail_u[i]]])
            x_pred = x_est.copy()
            collision = False
            x = []
            y = []
            for i in range(prediction_range):
                x_pred = self.motion_model(x_pred, u)
                x.append(x_pred[0][0])
                y.append(x_pred[1][0])
                if self.check_safe_collision(x_pred[0:2].reshape(2)):
                    collision = True
                    break
            if not collision:
                color = 'black'
                score = np.linalg.norm(self.x_goal - x_pred[0:2].reshape(2))
                if score < best_score:
                    best_u = u + 0.002 * np.random.rand() - 0.001  # TODO parametrize this noise
                    best_score = score
            else:
                # import ipdb
                # ipdb.set_trace()
                color = 'r'
            # plt.scatter(x, y, c=color)
        x_pred = x_est.copy()
        x = []
        y = []
        for i in range(prediction_range):
            x_pred = self.motion_model(x_pred, best_u)
            x.append(x_pred[0][0])
            y.append(x_pred[1][0])
        # plt.scatter(x, y, c='g')
        # plt.pause(0.001)
        # print("Best U: {}, Best Score: {}".format(best_u[1][0], best_score))
        return best_u

    def my_parallel_calc_input(self, x_est):
        # TODO work on it
        num_input_samples = 5
        prediction_range = 8
        v = self.v  # [m/s]
        yawrate = 0.00  # [rad/s]   default
        best_u = np.array([[v], [yawrate]])
        best_score = np.Inf
        avail_u = np.linspace(-math.pi / 2, math.pi / 2, num_input_samples)
        collision = np.tile(False, num_input_samples)
        states = np.tile(x_est, num_input_samples)
        vel = np.tile(v, num_input_samples)
        states_input = np.vstack([states, vel, avail_u])
        goal_states = np.tile(self.x_goal.reshape(2, 1), num_input_samples)
        for i in range(prediction_range):
            states_input = np.apply_along_axis(self.motion_model_array, 0, states_input)
            collision = collision | np.apply_along_axis(self.check_safe_collision_array, 0, states_input)
        score = np.linalg.norm(goal_states - states_input[:2, :], axis=0)
        import ipdb
        # ipdb.set_trace()
        if np.where(collision == False)[0].shape[0] > 0:
            best_score = np.min(score[np.where(collision == False)])
            i_best_u = np.where(score == best_score)[0][0]
            best_u = np.array([[v], [avail_u[i_best_u] + (0.02 * np.random.rand() - 0.01)]])
        return best_u

    def check_collision(self, xy):
        for coord in self.obstacle_coord:
            dist = np.linalg.norm(np.array([coord]) - xy)
            if dist <= (self.obstacle_radius + self.car_radius):
                return True
        return False

    def check_safe_collision(self, xy):
        for coord in self.obstacle_coord:
            dist = np.linalg.norm(np.array([coord]) - xy)
            if dist <= (self.obstacle_radius + self.car_radius + self.safe_margin):
                return True
        return False

    def goal_reached(self, xy):
        dist = np.linalg.norm(np.array(self.x_goal) - xy)
        if dist <= self.goal_radius:
            return True
        return False

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
        # Plot obstacles
        for x, y in self.obstacle_coord:
            ax.scatter(x, y, marker='+')
            ax.add_artist(plt.Circle((x, y), color='black', fill=True, radius=self.obstacle_radius))
            ax.add_artist(plt.Circle((x, y), color='r', fill=False, radius=self.obstacle_radius + self.car_radius))
            ax.add_artist(plt.Circle((x, y), color='b', fill=False,
                                     radius=self.obstacle_radius + self.car_radius + self.safe_margin))
        # Plot goal
        x = self.x_goal[0]
        y = self.x_goal[1]
        ax.add_artist(plt.Circle((x, y), color='g', fill=True, radius=self.goal_radius))
        ax.add_artist(plt.Circle((x, y), color='y', fill=False, radius=self.goal_radius + self.car_radius))
        if self.error_time is not None:
            true_value_coord = (self.hx_true[0, self.error_time].flatten(), self.hx_true[1, self.error_time].flatten())
            ax.scatter(self.hx_true[0, self.error_time].flatten(),
                       self.hx_true[1, self.error_time].flatten(), marker='X', c='r')  # mark the first error found
            ax.scatter(self.hx_est[0, self.error_time].flatten(),
                       self.hx_est[1, self.error_time].flatten(), marker='X', c='r')
            # ax.add_artist(plt.Circle(true_value_coord, color='r', fill=False, radius=self.ERR_THRESHOLD))
        else:
            # plot max reward
            true_value_coord = (
            self.hx_true[0, self.i_max_reward].flatten(), self.hx_true[1, self.i_max_reward].flatten())
            ax.scatter(self.hx_true[0, self.i_max_reward].flatten(),
                       self.hx_true[1, self.i_max_reward].flatten(), marker='+')
            ax.scatter(self.hx_est[0, self.i_max_reward].flatten(),
                       self.hx_est[1, self.i_max_reward].flatten(), marker='+')
            pass

        ax.axis("equal")
        ax.grid(True)
        ax.axis([-0.5, +9.5, -4, +2])  # just to fix the plot limits
        plt.pause(0.001)
        if save_fig:  # save only last plot
            fig_name = "{}_{}".format(prefix, strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
            plt.savefig(os.path.join(out_dir, fig_name))
