import numpy as np
import math
import matplotlib.pyplot as plt

class PFSystem:
    def __init__(self, time_horizon=10.0, transient_time=3.0, err_threshold=0.7, num_particles=100):
        # PF parameters
        self.Q = np.diag([0.2]) ** 2  # range error
        self.NP = num_particles     # number of particles
        self.NTh = self.NP / 2.0    # num particles for re-sampling
        #  Simulation parameter
        self.v = 1.5  # [m/s]
        self.GPS_NOISE = np.diag([0.5, 0.5]) ** 2
        self.INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
        self.DT = 0.1  # time tick [s]
        self.time_event = 5.0  # at this time, the system change orientation
        # Safety property parameters
        self.TIME_HORIZON = time_horizon   # simulation time [s]
        self.ERR_THRESHOLD = err_threshold  # threshold for error detection
        self.TRANSIENT_TIME = transient_time  # initial unstable period

        # State variables
        self.x_true = np.zeros((4, 1))
        self.x_est = np.zeros((4, 1))
        self.p_est = np.zeros((3, 3))
        self.time = 0.0
        self.original_s0 = np.vstack([np.zeros((4, 1)),  # x_true
                                      np.zeros((4, 1)),  # x_est
                                      np.zeros((3, 3)).reshape((9, 1)),  # p_est
                                      0.0])  # time
        # Particles
        self.px = np.zeros((4, self.NP))  # Particle state
        self.pw = np.zeros((1, self.NP)) + 1.0 / self.NP  # Particle weight
        # History
        self.hx_true = self.x_true
        self.hx_est = self.x_est
        self.hx_time = self.time
        self.hx_p_est = self.p_est.reshape((9, 1))  # flat p_est matrix
        self.hx_px = self.px.reshape(4*self.NP, 1)
        self.hx_pw = self.pw.reshape(self.NP, 1)

        # Rendering
        self.last_init_state = 0    # init state identified by index in trajectory
        self.i_max_reward = 0       # index of state with max reward
        self.robustness = np.Inf    # robustness value
        self.reward = 0             # reward value
        self.error_time = None

    def print_config(self):
        print("[Info] Environment: PF Localization")
        print("[Info] Parameters: Error Threshold {}, Time Horizon {}, Transient Time {}, Num Particles: {}[{}]".format(
                                                                                                self.ERR_THRESHOLD,
                                                                                                self.TIME_HORIZON,
                                                                                                self.TRANSIENT_TIME,
                                                                                                self.NP, self.NTh))
        print()

    def set_state(self, x_true, x_est, p_est, px, pw, time):
        # DEPRECATED, WORK ONLY WITH PREFIXES
        self.x_true = x_true
        self.x_est = x_est
        self.p_est = p_est
        self.time = time
        # Particles
        self.px = px
        self.pw = pw

    def set_prefix(self, sampled_prefix):
        # Set state
        self.x_true = sampled_prefix[0:4, -1].reshape((4, 1))       # 0,1,2,3 are true state
        self.x_est = sampled_prefix[4:8, -1].reshape((4, 1))        # 4,5,6,7 are est state
        self.p_est = sampled_prefix[8:17, -1].reshape((3, 3))       # 8,9,...16 are flatten cov matrix
        self.time = sampled_prefix[17:18, -1]                       # 17 is time
        self.px = sampled_prefix[18:18+4*self.NP, -1].reshape(4, self.NP)   # 18,...18+4*NP-1 are particles' states
        self.pw = sampled_prefix[18+4*self.NP:, -1].reshape(1, self.NP)        # 18+4*NP... are particles' weights
        # Set history
        self.hx_true = sampled_prefix[0:4, :]
        self.hx_est = sampled_prefix[4:8, :]
        self.hx_p_est = sampled_prefix[8:17, :]
        self.hx_time = sampled_prefix[17:18, :][0]                  # avoid dimension mismatch
        self.hx_px = sampled_prefix[18:18+4*self.NP, :]
        self.hx_pw = sampled_prefix[18+4*self.NP:, :]
        # Other info
        self.last_init_state = sampled_prefix.shape[1] - 1
        self.i_max_reward = sampled_prefix.shape[1] - 1
        self.robustness = np.Inf  # robustness value
        self.reward = 0
        self.error_time = None

    def reset_init_state(self):
        # State variables
        self.x_true = np.zeros((4, 1))
        self.x_est = np.zeros((4, 1))
        self.p_est = np.zeros((3, 3))
        self.time = 0.0
        # Particles
        self.px = np.zeros((4, self.NP))  # Particle store
        self.pw = np.zeros((1, self.NP)) + 1.0 / self.NP  # Particle weight
        # History
        self.hx_true = self.x_true
        self.hx_est = self.x_est
        self.hx_time = self.time
        self.hx_p_est = self.p_est.reshape((9, 1))  # flat p_est matrix
        self.hx_px = self.px.reshape((4 * self.NP, 1))  # flat p_est matrix
        self.hx_pw = self.pw.reshape((self.NP, 1))  # flat p_est matrix
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
        return np.vstack([self.x_true, self.x_est, self.p_est.reshape((9, 1)), self.time,
                          self.px.reshape(self.NP*4, 1), self.pw.reshape(self.NP, 1)])

    def get_trace(self):
        """
        Return the trace (trajectory) which leads to the current state.
        :return: trace of state variables
        """
        # IMPORTANT: the trace must contain also the info about the last particle swarm in order to restore the trace
        return np.vstack([self.hx_true, self.hx_est, self.hx_p_est, self.hx_time, self.hx_px, self.hx_pw])

    def is_current_trace_false(self):
        return self.error_time is not None

    def get_reward(self):
        return self.terminal_exp_reward(self.ERR_THRESHOLD, self.TRANSIENT_TIME)

    def terminal_exp_reward(self, ERR_THRESHOLD, TRANSIENT_TIME):
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
        :return: `isdone` if the trace is complete (time or succ), `rewar/FALSd`, eventual `error_time` in which error occurred
        """
        trans_minus_time = TRANSIENT_TIME - self.hx_time
        location_difference = np.linalg.norm(self.hx_true[0:2, :] - self.hx_est[0:2, :], axis=0)
        thresh_minus_err = ERR_THRESHOLD - location_difference
        max_array = np.max([trans_minus_time, thresh_minus_err], axis=0)
        i_min_rob = np.argmin(max_array[self.last_init_state:])     # only from the starting point
        best_rob_trace = max_array[self.last_init_state + i_min_rob]  # max because rob should be minimized

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
        self.i_max_reward = self.last_init_state + i_min_rob   # for rendering
        self.robustness = best_rob_trace
        return self.reward, self.error_time

    def run_system(self):
        while self.TIME_HORIZON > self.time:
            self.step_system()
            # store data history
            self.hx_true = np.hstack((self.hx_true, self.x_true))
            self.hx_est = np.hstack((self.hx_est, self.x_est))
            self.hx_p_est = np.hstack((self.hx_p_est, self.p_est.reshape(9, 1)))  # not used now
            self.hx_time = np.hstack((self.hx_time, self.time))    # all this to avoid dimension mistmatch
            self.hx_px = np.hstack((self.hx_px, self.px.reshape(4 * self.NP, 1)))
            self.hx_pw = np.hstack((self.hx_pw, self.pw.reshape(self.NP, 1)))
        self.get_reward()

    def step_system(self):
        self.time += self.DT
        u = self.my_calc_input(self.time)
        self.x_true, z, ud = self.observation(self.x_true, u)
        self.pf_parallel_localization(z, ud)
        #self.pf_localization(z, ud)

    def pf_localization(self, z, u):
        """
        Localization with Particle filter
        """
        for ip in range(self.NP):
            x = np.array([self.px[:, ip]]).T
            w = self.pw[0, ip]

            #  Predict with random input sampling
            ud = u + 1.5 * self.INPUT_NOISE @ np.random.randn(2, 1)
            x = self.motion_model(x, ud)

            #  Calc Importance Weight
            dx = x[0, 0] - z[0, 0]
            dy = x[1, 0] - z[1, 0]
            dz = math.sqrt(dx ** 2 + dy ** 2)
            w = w * self.gauss_likelihood(dz, math.sqrt(self.Q[0, 0]))

            self.px[:, ip] = x[:, 0]
            self.pw[0, ip] = w
        self.pw = self.pw / self.pw.sum()  # normalize

        self.x_est = self.px.dot(self.pw.T)
        self.p_est = self.calc_covariance(self.x_est)

        n_eff = 1.0 / (self.pw.dot(self.pw.T))[0, 0]  # Effective particle number
        if n_eff < self.NTh:
            self.re_sampling()

    def pf_parallel_localization(self, z, u):
        """
        Localization with Particle filter
        """
        # TODO work on it
        import ipdb
        ipdb.set_trace()
        inp = np.tile(u, self.NP)
        states_input = np.vstack([self.px, inp])

        for ip in range(self.NP):
            x = np.array([self.px[:, ip]]).T
            w = self.pw[0, ip]

            #  Predict with random input sampling
            ud = u + 1.5 * self.INPUT_NOISE @ np.random.randn(2, 1)
            x = self.motion_model(x, ud)

            #  Calc Importance Weight
            dx = x[0, 0] - z[0, 0]
            dy = x[1, 0] - z[1, 0]
            dz = math.sqrt(dx ** 2 + dy ** 2)
            w = w * self.gauss_likelihood(dz, math.sqrt(self.Q[0, 0]))

            self.px[:, ip] = x[:, 0]
            self.pw[0, ip] = w
        self.pw = self.pw / self.pw.sum()  # normalize

        self.x_est = self.px.dot(self.pw.T)
        self.p_est = self.calc_covariance(self.x_est)

        n_eff = 1.0 / (self.pw.dot(self.pw.T))[0, 0]  # Effective particle number
        if n_eff < self.NTh:
            self.re_sampling()

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

    @staticmethod
    def gauss_likelihood(x, sigma):
        p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
            math.exp(-x ** 2 / (2 * sigma ** 2))

        return p

    def calc_covariance(self, x_est):
        cov = np.zeros((3, 3))

        for i in range(self.px.shape[1]):
            dx = (self.px[:, i] - x_est)[0:3]
            cov += self.pw[0, i] * dx.dot(dx.T)
        cov /= self.NP

        return cov

    def re_sampling(self):
        """
        low variance re-sampling
        """
        w_cum = np.cumsum(self.pw)
        base = np.arange(0.0, 1.0, 1 / self.NP)
        re_sample_id = base + np.random.uniform(0, 1 / self.NP)
        indexes = []
        ind = 0
        for ip in range(self.NP):
            while re_sample_id[ip] > w_cum[ind]:
                ind += 1
            indexes.append(ind)
        # update particle states and weights
        self.px = self.px[:, indexes]
        self.pw = np.zeros((1, self.NP)) + 1.0 / self.NP  # init weight

    def observation(self, x_true, u):
        x_true = self.motion_model(x_true, u)

        # add noise to gps x-y
        z = self.observation_model(x_true) + self.GPS_NOISE @ np.random.randn(2, 1)

        # add noise to input
        ud = u + self.INPUT_NOISE @ np.random.randn(2, 1)

        return x_true, z, ud

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

    @staticmethod
    def observation_model(x):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = H @ x

        return z

    def my_calc_input(self, time):
        yaw_rate = 0.35  # [rad/s]
        if time >= self.time_event:
            yaw_rate = -0.35  # [rad/s]
        u = np.array([[self.v], [yaw_rate]])
        return u

    def render(self, title=""):
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
