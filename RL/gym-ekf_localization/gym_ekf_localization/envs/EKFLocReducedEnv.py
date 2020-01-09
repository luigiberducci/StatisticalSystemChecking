import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import math
import random as rand


#SECOND VERSION WITH OBS SPACE DEFINED BY SINGLE STATE
class EKFLocReducedEnv(gym.Env):
    def __init__(self):
        """ Define the action and observation spaces."""
        self.sys = System()
        DUMMY_ACTION = 1
        OBS_VARS = 25    #x,y,theta,v for real and approx state and time, and p_est flatten (4x4)
        epsilon = 0.5
        T0 = 0
        Tf = 10 + epsilon   #just to be sure
        # Safety condition
        self.ERR_THRESHOLD = 0.7  # threshold for error detection (if diff>eps then Error)
        self.TRANSIENT_TIME = 3.0  # initial transient time in which the localization is not stable
        self.TIME_HORIZON = 10.0  # initial transient time in which the localization is not stable
        # Env definition
        self.action_space = spaces.Discrete(DUMMY_ACTION)
        self.observation_space = spaces.Box(low=np.NINF,
                                            high=np.Inf,
                                            shape=(OBS_VARS, DUMMY_ACTION), dtype=np.float16)
        self.is_done = False
        self.reward = 0
        # Set of initial prefixes (IMPORTANCE SPLITTING)
        self.prefix_list = list()
        self.original_s0 = np.vstack([np.zeros((4, 1)),         # x_true
                                        np.zeros((4, 1)),       # x_est
                                        np.eye(4).reshape((16, 1)),  # p_est
                                        0.0])                   # time
        self.prefix_list.append(self.original_s0)
        # Init history for rendering
        self.best_rob_trace = np.Inf
        self.last_init_state = 0    # init state identified by index in trajectory
        self.error_time = None      # index of first state in which error occurred
        self.last_diff = 0          # for get_reward_0_1: give +1 if increase the error, 0 otherwise
        self.hx_true = self.sys.x_true
        self.hx_est = self.sys.x_est
        self.hx_time = self.sys.time
        self.hx_p_est = self.sys.p_est.reshape((16, 1))  #flat p_est matrix

    def print_info_config(self):
        print("[Info] Environment: EKF Localization")
        print("[Info] Parameters: Error Threshold {}".format(self.ERR_THRESHOLD))

    def get_state(self):
        """
        Return the current state (exec. trace) as a unique array (stack true, est, time arrays)
        :return: stacked representation of the current state
        """
        return np.vstack([self.sys.x_true, self.sys.x_est, self.sys.p_est.reshape((16, 1)), self.sys.time])

    def get_trace(self):
        """
        Return the trace (trajectory) which leads to the current state.
        :return: trace of state variables
        """
        return np.vstack([self.hx_true, self.hx_est, self.hx_p_est, self.hx_time])

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
        self.sys.step_system()  #Step run ahed
        self.is_done, self.reward, error_time = self.get_reward()
        if self.error_time is None and error_time is not None:         #Update only once (first state)
            self.error_time = error_time
        # store data history for rendering
        self.hx_true = np.hstack((self.hx_true, self.sys.x_true))
        self.hx_est = np.hstack((self.hx_est, self.sys.x_est))
        self.hx_p_est = np.hstack((self.hx_p_est, self.sys.p_est.reshape(16, 1)))  # not used now
        self.hx_time = np.hstack((self.hx_time, np.array([self.sys.time]).reshape(1,1)))    #all this to avoid dimension mistmatch

        return self.get_state(), self.reward, self.is_done, {}

    def sample_init_state(self):
        # `prefix_list`  is a list of prefixes, sample one of them uniformly at random
        sampled_prefix = rand.sample(self.prefix_list, 1)[0]  #sample return a list of 1 (k) element
        last_state = sampled_prefix.shape[1] - 1    #index of last state of prefix
        # Set state
        x_true = sampled_prefix[0:4, last_state].reshape((4, 1))
        x_est = sampled_prefix[4:8, last_state].reshape((4, 1))
        p_est = sampled_prefix[8:24, last_state].reshape((4, 4))
        time = sampled_prefix[24:25, last_state]
        # Set history
        self.hx_true = sampled_prefix[0:4, :]
        self.hx_est = sampled_prefix[4:8, :]
        self.hx_p_est = sampled_prefix[8:24, :]
        self.hx_time = sampled_prefix[24:25, :]
        self.error_time = None
        self.last_init_state = last_state
        # Copy is necessary otherwise the object in memory is the same and happened unexpected changes
        return x_true.copy(), x_est.copy(), p_est.copy(), time.copy()

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
        x_true, x_est, p_est, time = self.sample_init_state()
        self.last_diff = 0          # for get_reward_0_1: give +1 if increase the error, 0 otherwise
        self.sys.set_state(x_true, x_est, p_est, time)
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
        ax.plot(self.hx_true[0, :].flatten(),
                self.hx_true[1, :].flatten(), "-g")  # plot the true trajectory
        ax.plot(self.hx_est[0, :].flatten(),
                self.hx_est[1, :].flatten(), "-b")  # plot the estimated (belief) trajectory
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
            ax.add_artist(plt.Circle(true_value_coord, color='r', fill=False, radius=0.7))

        ax.axis("equal")
        ax.grid(True)
        plt.pause(0.001)

    def get_reward(self):
        return self.terminal_exp_reward(self.ERR_THRESHOLD, self.TRANSIENT_TIME, self.TIME_HORIZON)

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
        trans_minus_time = (TRANSIENT_TIME - self.sys.time[0])
        location_difference = np.linalg.norm(self.sys.x_true[0:2] - self.sys.x_est[0:2], axis=0)[0]
        thresh_minus_err = ERR_THRESHOLD - location_difference
        self.best_rob_trace = np.min([self.best_rob_trace, np.max([trans_minus_time, thresh_minus_err])])  #max because rob should be minimized

        # Flag for falsification condition
        error_found = True if self.sys.time > TRANSIENT_TIME and location_difference > ERR_THRESHOLD else False
        time_elapsed = True if self.sys.time >= TIME_HORIZON else False
        error_time = None
        if self.error_time is None:
            error_time = None if not error_found else self.hx_true.shape[1]  # Notice: not shape-1 because the errortime is the current (not yet appended)
        is_done = time_elapsed
        reward = 0
        if is_done:
            reward = np.exp(-self.best_rob_trace)
        if reward>1 and (self.error_time is None and error_time is None):
            import ipdb
            ipdb.set_trace()
        return is_done, reward, error_time

    def new_positive_reward(self, ERR_THRESHOLD, TRANSIENT_TIME, TIME_HORIZON):
        """
        Reward defined on formula p = m>=0 | -p | p or q
        as follow:
            reward(m>=0) = 1/eval(m)
            reward(-p) = -reward(p)
            reward(p or q) = min(reward(p), reward(q))
        where eval(m) = m if m>0, otherwise +inf
        :param ERR_THRESHOLD:   threshold for error detection
        :param TRANSIENT_TIME: time when finish the transient phase in which the localization is unstable
        :param TIME_HORIZON: total lenght of trace
        :return: `isdone` if the trace is complete (time or succ), `reward`, eventual `error_time` in which error occurred
        """
        SUCC_REWARD = +1000  # reward for success

        trans_minus_time = 1 / ((TRANSIENT_TIME - self.sys.time[
            0]) / TIME_HORIZON) if TRANSIENT_TIME - self.sys.time >= 0 else SUCC_REWARD
        location_difference = np.linalg.norm(self.sys.x_true[0:2] - self.sys.x_est[0:2], axis=0)[0]
        thresh_minus_err = 1 / (
                    ERR_THRESHOLD - location_difference) if ERR_THRESHOLD - location_difference >= 0 else SUCC_REWARD
        reward = min(trans_minus_time, thresh_minus_err)

        # Flag for falsification condition
        error_found = True if self.sys.time > TRANSIENT_TIME and location_difference > ERR_THRESHOLD else False
        time_elapsed = True if self.sys.time > TIME_HORIZON else False
        error_time = None if not error_found else self.hx_true.shape[
            1]  # Notice: not shape-1 because the errortime is the current (not yet appended)
        is_done = error_found or time_elapsed
        return is_done, reward, error_time

    def new_exp_positive_reward(self, ERR_THRESHOLD, TRANSIENT_TIME, TIME_HORIZON):
        """
        Reward defined on formula p = m>=0 | -p | p or q
        as reward(phi) = exp(-rob(phi)) - 1, where rob is
            rob(m>=0) = exp(-eval(m)) - 1
            rob(-p) = -rob(p)
            rob(p or q) = max(rob(p), rob(q))
        where eval(m) = m
        :param ERR_THRESHOLD:   threshold for error detection
        :param TRANSIENT_TIME: time when finish the transient phase in which the localization is unstable
        :param TIME_HORIZON: total lenght of trace
        :return: `isdone` if the trace is complete (time or succ), `reward`, eventual `error_time` in which error occurred
        """
        trans_minus_time = (TRANSIENT_TIME - self.sys.time[0]) / TIME_HORIZON
        location_difference = np.linalg.norm(self.sys.x_true[0:2] - self.sys.x_est[0:2], axis=0)[0]
        thresh_minus_err = ERR_THRESHOLD - location_difference
        rob_phi = np.max([trans_minus_time, thresh_minus_err])  #max because rob should be minimized
        reward = np.exp(-rob_phi)

        # Flag for falsification condition
        error_found = True if self.sys.time > TRANSIENT_TIME and location_difference > ERR_THRESHOLD else False
        time_elapsed = True if self.sys.time > TIME_HORIZON else False
        error_time = None if not error_found else self.hx_true.shape[1]  # Notice: not shape-1 because the errortime is the current (not yet appended)
        is_done = error_found or time_elapsed
        return is_done, reward, error_time

    def positive_reward(self, ERR_THRESHOLD, TRANSIENT_TIME, TIME_HORIZON):
        """ GOOD PERFORMANCE """
        ERROR_REWARD = +10  # reward if success (error found)

        location_differences = np.linalg.norm(self.sys.x_true[0:2] - self.sys.x_est[0:2], axis=0)[0]
        # Flag for falsification condition
        error_found = True if self.sys.time > TRANSIENT_TIME and location_differences > ERR_THRESHOLD else False
        time_elapsed = True if self.sys.time > TIME_HORIZON else False
        error_time = None if not error_found else self.hx_true.shape[1]
        is_done = error_found or time_elapsed
        if self.sys.time <= TRANSIENT_TIME :
            reward = -1
        elif not error_found:
            reward = location_differences
        else:
            reward = ERROR_REWARD
        return is_done, reward, error_time

    def negative_reward(self, ERR_THRESHOLD, TRANSIENT_TIME, TIME_HORIZON):
        """ PROBLEM: cannot be minimization because of return definition as sum of reward
            Then QL always prefer terminal states, not good
            Based on robustness estimation on current state (no future view) """
        ERROR_REWARD = -10  # reward if success (error found)
        # Given spec phi:= G(p or q) wt p:=error<=threshold, q:=time<=transient
        # p = threshold >= err -> rob(p) = threshold-err
        location_difference = ERR_THRESHOLD - np.linalg.norm(self.sys.x_true[0:2] - self.sys.x_est[0:2], axis=0)[0]
        # q = time <= transient -> rob(q) = transient-time
        time_diff = (TRANSIENT_TIME - self.sys.time)[0]
        # reward as rob(spec) = max(rob(p), rob(q))
        reward = max(time_diff, location_difference)
        # Flag for falsification condition
        error_found = reward < 0
        time_elapsed = True if self.sys.time > TIME_HORIZON else False
        error_time = None if not error_found else self.hx_true.shape[1]
        is_done = error_found or time_elapsed
        reward = reward if not error_found else ERROR_REWARD
        return is_done, reward, error_time

class System:
    def __init__(self):
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
        self.SIM_TIME = 10.0  # simulation time [s]
        self.time_event = 5.0  # at this time, the system change orientation

        self._num_sim_steps = math.ceil((self.SIM_TIME + self.DT) / self.DT + 1)    #+1 for 0 time

        #State variables
        self.x_true = np.zeros((4, 1))
        self.x_est = np.zeros((4, 1))
        self.p_est = np.eye(4)
        self.time = 0.0

    def reset_init_state(self):
        self.x_true = np.zeros((4, 1))
        self.x_est = np.zeros((4, 1))
        self.p_est = np.eye(4)
        self.time = 0.0

    def set_state(self, x_true, x_est, p_est, time):
        self.x_true = x_true
        self.x_est = x_est
        self.p_est = p_est
        self.time = time

    def run_system(self):
        hx_est = self.x_est
        hx_true = self.x_true
        hx_time = self.time
        hx_p_est = self.p_est

        while self.SIM_TIME > self.time:
            self.step_system()
            # store data history
            hx_true = np.hstack((hx_true, self.x_true))
            hx_est = np.hstack((hx_est, self.x_est))
            hx_p_est = np.hstack((hx_p_est, self.p_est))    #not used now
            hx_time = np.hstack((hx_time, self.time))

        return hx_true, hx_est, hx_time

    def step_system(self):
        self.time += self.DT
        u = self.my_calc_input(self.time)
        x_dr = self.x_true      # to maintain previous method
        self.x_true, z, xDR, ud = self.observation(self.x_true, x_dr, u)
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


