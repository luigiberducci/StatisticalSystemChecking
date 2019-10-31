import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import math

class EKFLocEnv(gym.Env):
    def __init__(self):
        """ Define the action and observation spaces."""
        self.sys = System()
        self.mon = Monitor()
        TRACE_LEN = self.sys.get_number_of_steps()
        OBS_VARS = self.sys.get_number_of_observed_vars()
        self.action_space = spaces.Discrete(TRACE_LEN)
        # TODO: find a way to bound the observation space
        self.observation_space = spaces.Box(low=-np.Inf, high=np.NINF, shape=(OBS_VARS, TRACE_LEN), dtype=np.float16)

    def get_state(self):
        """
        Return the current state (exec. trace) as a unique array (stack true, est, time arrays)
        :return: stacked representation of the current state
        """
        return np.vstack([self.true[0:2, :], self.est[0:2, :], self.time])

    def step(self, action_prefix):
        """
        Starting from the current state, create a break-point according to the index `action_prefix`
        Then re-run the system from that state and update the current state, as merging of:
            -prefix: current state up to the break-point (exclude break-point)
            -suffix: new run from the break-point (included) to the time horizon
        :param action_prefix: index in [0, len(trace)) in which create the breakpoint
        :return: state, reward, id_done flag, and info
        """
        if self.is_done:    #avoid to loose a starting good state
            return self.get_state(), self.reward, self.is_done, {}
        if action_prefix <= self.last_action:    #avoid to backtrack
            return self.get_state(), 0, self.is_done, {}

        # Compute prefix (unchanged)
        true_prefix = self.true[:, 0:action_prefix] #the index `action_prefix` is excluded
        est_prefix = self.est[:, 0:action_prefix]

        # State from which start a new run
        true_state = np.reshape(self.true[:, action_prefix], (4, 1))
        est_state = np.reshape(self.est[:, action_prefix], (4,1))
        curr_time = self.time[action_prefix]

        # Merge the two traces
        true_suffix, est_suffix, time = self.sys.run_system_from_state(true_state, est_state, curr_time)
        self.true = np.concatenate((true_prefix, true_suffix), axis=1)
        self.est = np.concatenate((est_prefix, est_suffix), axis=1)
        # time not changed
        self.last_action = action_prefix

        # Compute reward in the new trace
        self.is_done, self.reward, self.error_time = self.mon.sat_property(self.true, self.est, self.time)
        if self.is_done:
            self.reward = 10

        return self.get_state(), self.reward, self.is_done, {}

    def reset(self):
        """
        Reset the state of the problem, creating a new complete trace.
        :return: initial state
        """
        self.true, self.est, self.time = self.sys.run_system()
        self.is_done, self.reward, self.error_time = self.mon.sat_property(self.true, self.est, self.time)
        if self.is_done:
            self.reward = 10
        self.last_action = 0
        return self.get_state()

    def render(self, mode='human'):
        """
        Render method to show the current state (execution trace) of the problem.
        It plots the trajectory of the system: the true trajectory and the belief trajectory.
        :return: -
        """
        ax = plt.gca()
        ax.cla()
        ax.plot(self.true[0, :].flatten(),
                self.true[1, :].flatten(), "-g")  # plot the true trajectory
        ax.plot(self.est[0, :].flatten(),
                self.est[1, :].flatten(), "-b")  # plot the estimated (belief) trajectory
        ax.scatter(self.true[0, self.last_action].flatten(),
                   self.true[1, self.last_action].flatten(), marker='o')  # mark last break-point (action)
        ax.scatter(self.est[0, self.last_action].flatten(),
                   self.est[1, self.last_action].flatten(), marker='o')
        if self.error_time is not None:
            true_value_coord = (self.true[0, self.error_time].flatten(), self.true[1, self.error_time].flatten())
            ax.scatter(self.true[0, self.error_time].flatten(),
                       self.true[1, self.error_time].flatten(), marker='X')  # mark the first error found
            ax.scatter(self.est[0, self.error_time].flatten(),
                       self.est[1, self.error_time].flatten(), marker='X')
            ax.add_artist(plt.Circle(true_value_coord, color='r', fill=False, radius=0.6))

        ax.axis("equal")
        ax.grid(True)
        plt.pause(0.001)

    def holdon_plot(self, mode='human'):
        """
        Render method to show the current state (execution trace) of the problem.
        It plots the trajectory of the system: the true trajectory and the belief trajectory.
        :return: -
        """
        ax = plt.gca()
        ax.cla()
        ax.plot(self.true[0, :].flatten(),
                 self.true[1, :].flatten(), "-g")   # plot the true trajectory
        ax.plot(self.est[0, :].flatten(),
                 self.est[1, :].flatten(), "-b")    # plot the estimated (belief) trajectory
        ax.scatter(self.true[0, self.last_action].flatten(),
                    self.true[1, self.last_action].flatten(), marker='o')   # mark last break-point (action)
        ax.scatter(self.est[0, self.last_action].flatten(),
                    self.est[1, self.last_action].flatten(), marker='o')
        if self.error_time is not None:
            true_value_coord = (self.true[0, self.error_time].flatten(), self.true[1, self.error_time].flatten())
            ax.scatter(self.true[0, self.error_time].flatten(),
                        self.true[1, self.error_time].flatten(), marker='X')  # mark the first error found
            ax.scatter(self.est[0, self.error_time].flatten(),
                        self.est[1, self.error_time].flatten(), marker='X')
            ax.add_artist(plt.Circle(true_value_coord, color='r', fill=False, radius=0.6))

        ax.axis("equal")
        ax.grid(True)
        plt.show()

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
        self._num_obs_vars  = 5  #obs vars: x,y of true trajectory, x,y of estimated, and time

    def get_number_of_steps(self):
        return self._num_sim_steps

    def get_number_of_observed_vars(self):
        return self._num_obs_vars

    def run_system(self):
        time = 0.0
        # State Vector [x y yaw v]'
        x_est = np.zeros((4, 1))
        x_true = np.zeros((4, 1))
        return self.run_system_from_state(x_true, x_est, time)

    def run_system_from_state(self, xTrue, xEst, time):
        xDR = xTrue
        hxEst = xEst
        hxTrue = xTrue

        hxTime = time
        hz = np.zeros((2, 1))
        PEst = np.eye(4)

        while self.SIM_TIME > time:
            time += self.DT
            u = self.my_calc_input(time)

            xTrue, z, xDR, ud = self.observation(xTrue, xDR, u)

            xEst, PEst = self.ekf_estimation(xEst, PEst, z, ud)

            # store data history
            hxEst = np.hstack((hxEst, xEst))
            hxTime = np.hstack((hxTime, time))
            hxTrue = np.hstack((hxTrue, xTrue))
            hz = np.hstack((hz, z))

        return hxTrue, hxEst, hxTime

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

class Monitor:
    """ Monitor for the temporal property.
        Property:   Eventually error in the estimate is greater than `epsilon`
                    (F time>transient & |true-estimate|>epsilon)
        epsilon = 0.6 -> 9/1000 reach error state
    """
    def __init__(self):
        self.EPSILON = 0.5          # threshold for error detection (if diff>eps then Error)
        self.TRANSIENT_TIME = 3.0   # initial transient time in which the localization is not stable

    def sat_property(self, true, est, time):
        location_differences = np.linalg.norm(true[0:2, :] - est[0:2, :], axis=0)
        non_zero = np.argwhere((time > self.TRANSIENT_TIME) & (
                                location_differences > self.EPSILON))
        monitor = (len(non_zero)>0)
        error_time = None if len(non_zero) == 0 else non_zero[0]
        max_difference = max(location_differences)
        return monitor, max_difference, error_time