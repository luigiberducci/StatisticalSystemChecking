import os
import random as rand
from time import strftime, gmtime

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

car1_imagefile = mpimg.imread("/home/luigi/Development/StatisticalSystemChecking/RLIS/env/car1_color.png")
car2_imagefile = mpimg.imread("/home/luigi/Development/StatisticalSystemChecking/RLIS/env/car2_color.png")
car1_image = OffsetImage(car1_imagefile, zoom=0.05)
car2_image = OffsetImage(car2_imagefile, zoom=0.05)

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


class LinearCar:
    """
    Car system
    """
    def __init__(self, x0, y0, theta, fullranges, v1_range_id, v2_range_id, v3_range_id):
        # Fixed Parameters
        self.road_segment1 = 20
        self.road_segment2 = 50
        self.road_segment3 = 70
        self.theta = theta  # fixed orientation of car
        # Variable Parameters
        self.fullranges = fullranges
        self.v1range = self.fullranges[v1_range_id]
        self.v2range = self.fullranges[v2_range_id]
        self.v3range = self.fullranges[v3_range_id]
        # State variables
        self.x0 = x0
        self.y0 = y0
        self.x = x0
        self.y = y0
        self.t = 0
        self.v = 0

    def set_state(self, x, y, t, v1_range_id, v2_range_id, v3_range_id):
        # Variable Parameters
        self.v1range = self.fullranges[v1_range_id]
        self.v2range = self.fullranges[v2_range_id]
        self.v3range = self.fullranges[v3_range_id]
        # State variables
        self.x = x
        self.y = y
        self.v = 0
        self.t = t

    def get_state(self):
        return np.array([self.x, self.y, self.v]).reshape(3, 1)

    def step(self, dt):
        v = self.calc_input()
        self.x = self.x + dt * math.cos(self.theta) * v
        self.y = self.y + dt * math.sin(self.theta) * v
        self.v = v
        self.t += dt

    def calc_input(self):
        passed_road = np.linalg.norm(np.array([self.x, self.y]) - np.array([self.x0, self.y0]))
        if passed_road <= self.road_segment1:
            start, stop = self.v1range
        elif passed_road <= self.road_segment2:
            start, stop = self.v2range
        elif passed_road <= self.road_segment3:
            start, stop = self.v3range
        else:
            start, stop = [0, 0]  # stop if too long simulation
        return rand.uniform(start, stop)


class CrossRoadEnv:
    """
    Environment Crossing road with 2 cars
    """

    def __init__(self):
        # environment full ranges
        self.v_ranges = [[0, 1.5], [1.51, 3], [3.01, 5]]
        self.T = 15
        self.dt = 0.1
        self.collision_range = 2.0
        # define random ranges
        i1, i2, i3 = 0, 0, 0
        j1, j2, j3 = 0, 0, 0
        # instanciate car systems
        self.car1 = LinearCar(55, -5, math.radians(90), self.v_ranges, i1, i2, i3)
        self.car2 = LinearCar(-5, 10, 0, self.v_ranges, j1, j2, j3)
        # env state
        self.car1_coord = self.car1.get_state()
        self.car2_coord = self.car2.get_state()
        self.car1_v_range = np.array([i1, i2, i3]).reshape(3, 1)
        self.car2_v_range = np.array([j1, j2, j3]).reshape(3, 1)
        self.t = 0
        self.original_s0 = None
        # env history
        self.hx_car1_coord = self.car1_coord
        self.hx_car2_coord = self.car2_coord
        self.hx_car1_v_range = self.car1_v_range
        self.hx_car2_v_range = self.car2_v_range
        self.hx_time = self.t
        # rendering
        self.error_time = None
        self.last_init_state = 0
        self.i_max_reward = 0
        self.reward = 0
        self.reward_array = None
        self.robustness = np.Inf  # robustness value
        self.rob_avg = 1 / 2  # mean rob for Robustness Scaling, init 1/2 means that doesn't alter the result by default

    def reset_init_state(self):
        # define random ranges
        i1, i2, i3 = [rand.randint(0, len(self.v_ranges) - 1) for _ in range(3)]
        j1, j2, j3 = [rand.randint(0, len(self.v_ranges) - 1) for _ in range(3)]
        # instanciate car systems
        self.car1 = LinearCar(55, -5, math.radians(90), self.v_ranges, i1, i2, i3)
        self.car2 = LinearCar(-5, 10, 0, self.v_ranges, j1, j2, j3)
        # env state
        self.car1_coord = self.car1.get_state()
        self.car2_coord = self.car2.get_state()
        self.car1_v_range = np.array([i1, i2, i3]).reshape(3, 1)
        self.car2_v_range = np.array([j1, j2, j3]).reshape(3, 1)
        self.t = 0
        # env history
        self.hx_car1_coord = self.car1_coord
        self.hx_car2_coord = self.car2_coord
        self.hx_car1_v_range = self.car1_v_range
        self.hx_car2_v_range = self.car2_v_range
        self.hx_time = self.t
        # rendering
        self.error_time = None
        self.reward = 0
        self.reward_array = None
        self.robustness = np.Inf  # robustness value
        self.last_init_state = 0

    def step_system(self):
        # assume first car1, then car2
        self.car1.step(self.dt)
        self.car2.step(self.dt)
        # update state
        self.car1_coord = self.car1.get_state()
        self.car2_coord = self.car2.get_state()

    def run_system(self):
        while self.t < self.T:
            if self.check_collision():
                break
            self.step_system()
            self.t += self.dt
            # store data history
            self.hx_car1_coord = np.hstack((self.hx_car1_coord, self.car1_coord))
            self.hx_car2_coord = np.hstack((self.hx_car2_coord, self.car2_coord))
            self.hx_car1_v_range = np.hstack((self.hx_car1_v_range, self.car1_v_range))
            self.hx_car2_v_range = np.hstack((self.hx_car2_v_range, self.car2_v_range))
            self.hx_time = np.hstack((self.hx_time, self.t))  # all this to avoid dimension mistmatch
        self.get_reward()

    def print_config(self):
        print("[Info] Environment: Crossing Road")
        print("[Info] Parameters: 2 Cars, Velocity ranges changes at 5, 10, 15 secs. Full velocity range [0,5]")
        print()

    def set_rob_scaling(self, rob_avg):
        """
        Set the mean min robustness value for Robustness Scaling.
        :param rob_avg:     estimated min robustness
        """
        self.rob_avg = rob_avg

    def set_prefix(self, prefix):
        # set state
        #import ipdb
        #ipdb.set_trace()
        self.car1_coord = prefix[0:3, -1]
        self.car2_coord = prefix[3:6, -1]
        self.car1_v_range = prefix[6:9, -1].reshape(3, 1)
        self.car2_v_range = prefix[9:12, -1].reshape(3, 1)
        self.t = prefix[12, -1]
        # set car state
        x, y, t = self.car1_coord[0], self.car1_coord[1], self.t
        range1, range2, range3 = self.car1_v_range
        self.car1.set_state(x, y, t, int(range1[0]), int(range2[0]), int(range3[0]))
        x, y, t = self.car2_coord[0], self.car2_coord[1], self.t
        range1, range2, range3 = self.car2_v_range
        self.car2.set_state(x, y, t, int(range1[0]), int(range2[0]), int(range3[0]))
        # set history
        self.hx_car1_coord = prefix[0:3, :]
        self.hx_car2_coord = prefix[3:6, :]
        self.hx_car1_v_range = prefix[6:9, :]
        self.hx_car2_v_range = prefix[9:12, :]
        self.hx_time = prefix[12, :]
        # Rendering
        self.last_init_state = prefix.shape[1] - 1
        self.i_max_reward = prefix.shape[1] - 1
        self.robustness = np.Inf  # robustness value
        self.reward = 0
        self.error_time = None

    def get_trace(self):
        return np.vstack([self.hx_car1_coord,
                          self.hx_car2_coord,
                          self.hx_car1_v_range,
                          self.hx_car2_v_range,
                          self.hx_time])

    def is_current_trace_false(self):
        return self.error_time is not None

    def get_reward(self):
        return self.compute_mc_reward_array()

    def check_collision(self):
        dist = np.linalg.norm(self.car1_coord[:2] - self.car2_coord[:2])
        return dist <= self.collision_range

    def compute_mc_reward_array(self):
        """
        Compute the `reward_array` R of the current trace, defined inductively as follows:
            R_t = exp(-rho_t)
            R_i = exp(-min{ rho_j | j>=i })     # future min
        """
        dist = np.linalg.norm(self.hx_car1_coord[0:2, :] - self.hx_car2_coord[0:2, :], axis=0)
        rob_array = dist - self.collision_range
        self.i_max_reward = np.argmin(rob_array)
        self.robustness = rob_array[self.i_max_reward]
        # Novelty here: compute future min robustness for each state in the trace
        future_rob_array = get_future_minimum_array(rob_array)
        # Robustness scaling
        norm_rob_array = future_rob_array / (2 * self.rob_avg)
        reward_array = np.exp(-norm_rob_array)
        # Error detection
        error_mask = dist < self.collision_range
        error_found = any(error_mask)
        if error_found:
            self.error_time = np.where(error_mask)[0][0]
            self.render()
            #import ipdb
            #ipdb.set_trace()
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
        dist = np.linalg.norm(self.car1_coord[0:2, :] - self.car2_coord[0:2, :], axis=0)
        rob_array = dist - self.collision_range
        i_min_robustness = np.argmin(rob_array)
        robustness = rob_array[i_min_robustness]
        return i_min_robustness, robustness

    def render(self, title="", save_fig=False, out_dir="out", prefix="falsification"):
        """
        Render method to show the current state.
        It plots the trajectories of the cars
        :return: -
        """
        fig = plt.figure(1)
        ax = plt.gca()
        ax.cla()
        ax.set_title(title)
        ax.set_xlim(-10, 70)
        ax.set_ylim(-10, 70)
        ax.axvline(x=55, linewidth=20.0, color="grey", alpha=0.4)
        ax.axhline(y=10, linewidth=20.0, color="grey", alpha=0.4)
        ax.plot()
        #lc = mc.LineCollection(self.hx_car1_coord[0:2, :], cmap='viridis')
        ax.plot(self.hx_car1_coord[0, :].flatten(),
                self.hx_car1_coord[1, :].flatten(), "-g", label="Car1")  # plot the true trajectory
        ax.plot(self.hx_car2_coord[0, :].flatten(),
                self.hx_car2_coord[1, :].flatten(), "-b", label="Car2")  # plot the estimated (belief) trajectory
        ax.scatter(self.hx_car1_coord[0, :].flatten(), self.hx_car1_coord[1, :].flatten(),
                   c=self.hx_car1_coord[2, :].flatten())  # plot the true trajectory
        ax.scatter(self.hx_car2_coord[0, :].flatten(), self.hx_car2_coord[1, :].flatten(),
                   c=self.hx_car2_coord[2, :].flatten())  # plot the estimated (belief) trajectory
        ax.legend()
        ax.scatter(self.hx_car1_coord[0, self.last_init_state].flatten(),
                   self.hx_car1_coord[1, self.last_init_state].flatten(), marker='o')  # mark last break-point (action)
        ax.scatter(self.hx_car2_coord[0, self.last_init_state].flatten(),
                   self.hx_car2_coord[1, self.last_init_state].flatten(), marker='o')
        #fig.colorbar(im, ax=ax)
        car1_coord = (self.hx_car1_coord[0, -1].flatten(), self.hx_car1_coord[1, -1].flatten()-1)
        car2_coord = (self.hx_car2_coord[0, -1].flatten()-0.5, self.hx_car2_coord[1, -1].flatten())
        ax.add_artist(AnnotationBbox(car1_image, car1_coord, frameon=False))
        ax.add_artist(AnnotationBbox(car2_image, car2_coord, frameon=False))
        if self.error_time is not None:
            car1_coord = (
            self.hx_car1_coord[0, self.error_time].flatten(), self.hx_car1_coord[1, self.error_time].flatten())
            ax.scatter(self.hx_car1_coord[0, self.error_time].flatten(),
                       self.hx_car1_coord[1, self.error_time].flatten(), marker='X')  # mark the first error found
            ax.scatter(self.hx_car2_coord[0, self.error_time].flatten(),
                       self.hx_car2_coord[1, self.error_time].flatten(), marker='X')
            ax.add_artist(plt.Circle(car1_coord, color='r', fill=False, radius=self.collision_range))
        else:
            # plot max reward
            ax.scatter(self.hx_car1_coord[0, self.i_max_reward].flatten(),
                       self.hx_car1_coord[1, self.i_max_reward].flatten(), marker='+')  # mark the first error found
            ax.scatter(self.hx_car2_coord[0, self.i_max_reward].flatten(),
                       self.hx_car2_coord[1, self.i_max_reward].flatten(), marker='+')
        #ax.axis("equal")
        #ax.grid(True)
        plt.pause(0.001)
        if save_fig:  # save only last plot
            fig_name = "{}_{}".format(prefix, strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
            plt.savefig(os.path.join(out_dir, fig_name))
