from keras.callbacks import Callback
import collections
import os
from time import gmtime, strftime
import numpy as np
import matplotlib.pyplot as plt

class ImportanceSplittingCallback(Callback):
    def __init__(self, env, agent, num_particles=100, k=10, warmup_skip=50, run_ispl=True):
        self.env = env  #Store the environment for prefix interaction
        self.agent = agent  #Store the agent for later using of the qnet
        self.warmup_episodes = warmup_skip  #iterations to skip to allow the qvalue to have sense in the first states
        self.use_i_splitting = run_ispl  #Flag to run isplit or simple monte carlo simulations
        # Debug options
        self.falsification_counter = 0  #no reset
        self.step_by_step_exec = False  # enable input at each new level, reset or falsification
        # Configure ISplit
        self.num_particles = num_particles  # Number of particles for ISplit
        self.k_best_level = k               # Number of particles for next level computation
        self.omega = list()     # List of trace of states
        self.q_omega = list()   # List of trace of QValue of each state
        self.s_omega = list()   # List of score for each trace
        self.level = np.NINF     # Current level, since minimization initialize as INF
        self.level_counter = 0    #current number of levels
        self.time_advance_range = 10    #each level consider further in time
        # Storing level info
        self.level_list = []        # list of level in current execution (each new restart, reset)
        self.level_prob_list = []   # list of level reach. prob in current exec (each restart, reset)
        self.error_prob_list = []   # list of falsification probability of all execution (permanent)
        # Initialize indices for writing in `omega`
        self.available_indices = collections.deque(maxlen=self.num_particles)
        for i in range(self.num_particles):     #init: all indices available
            self.available_indices.append(i)

    def reset_importance_splitting(self):
        # Reset env prefix from the original one
        self.env.reset_original_prefix_list()
        # All data structures empty
        self.omega = list()  # List of trace of states
        self.q_omega = list()  # List of trace of QValue of each state
        self.s_omega = list()  # List of score for each trace
        self.level = np.NINF  # Current level, since minimization initialize as INF
        self.level_counter = 0    #current number of levels
        # Rendering of levels
        self.level_list = []
        self.level_prob_list = []   # list of level reach. prob in current exec (each restart, reset)
        # Initialize indices for writing in `omega`
        self.available_indices.clear()
        for i in range(self.num_particles):  # init: all indices available
            self.available_indices.append(i)

    def on_episode_end(self, episode, logs={}):
        if not self.use_i_splitting:        #disable ispl, run only MC+Qlearning (no exploration policy)
            return
        if episode < self.warmup_episodes:
            return
        #self.plot_episode()

        if self.available_indices:              # Not empty
            last_i = self.store_current_trace()
            if self.env.is_current_trace_false():   # Error found
                self.falsification_procedure(last_i)      # Plot, Compute probability, ...
                self.reset_importance_splitting()   # Reset ISplit from scratch
                return  #Note: if no reset, search stucked on error trace, produce always the same
        if not self.available_indices:  # Empty = We collected all the particles required
            # Note: don't use `else` otherwise we loose an iteration
            self.run_importance_splitting()

    def store_current_trace(self):
        i = self.available_indices.popleft()  # get the first available index in omega
        assert i <= len(self.omega)  # at most is the new element to insert
        if len(self.omega) > i:
            self.omega[i] = self.env.get_trace()  # store the current trace (episode)
        else:
            self.omega.append(self.env.get_trace())  # store the current trace (episode)
        return i    #return position of last writing

    def run_importance_splitting(self):
        assert not self.available_indices           #we collected all the particles
        self.evaluate_all_traces_with_qlearning()   #for each trace, compute q values and score (min qval in trace)
        found_next_level, self.level = self.choose_next_level()
        if not found_next_level:        # If not found next better level with K particles, restart all
            print("[Info] NOT FOUND NEW LEVEL, Current Best Level is {}".format(self.level))
            if self.step_by_step_exec:
                input()
            self.reset_importance_splitting()   #Restart ISplit from scratch
            return
        bool_split = self.split_traces(self.level)  # check each trace if is above or below the current level
        self.env.clear_prefix_list()    # once computed the new level, reset the prefix list to empty
        k_prom_traces = 0
        k_non_prom_traces = 0
        for i, b in enumerate(bool_split):
            if b == True:
                # Promising trace, compute prefix and add it to env as restarting point
                k_prom_traces = k_prom_traces + 1
                prefix = self.compute_prefix_of_trace(i, self.level)
                self.env.add_prefix(prefix)
            else:
                # Non-promising trace, flag it for next replacement
                k_non_prom_traces = k_non_prom_traces + 1
                self.available_indices.append(i)    # trace `i` will be overwritten
        self.level_counter = self.level_counter + 1
        print("[Info] {}-TH LEVEL FOUND: {}, Considering {}/{} traces".format(self.level_counter, self.level, k_prom_traces, k_non_prom_traces))
        # Store level info
        self.level_list.append(self.level)
        self.level_prob_list.append(k_prom_traces/(k_prom_traces+k_non_prom_traces))    #Compute prob as avg
        self.plot_score()
        if self.step_by_step_exec:
            input()

    def evaluate_all_traces_with_qlearning(self):
        """
        Evaluation of the current set of traces (particles) using Q Network.
        This method fills `q_omega` and `s_omega`.
        :return: -
        """
        for i, trace in enumerate(self.omega):
            q_trace, score = self.compute_q_trace_and_score_of_trace(i)   # qtrace of qval for each state, score computed by them
            assert i <= len(self.q_omega)       # at most is the new element to insert
            if len(self.q_omega) > i:
                self.q_omega[i] = q_trace
                self.s_omega[i] = score
            else:
                self.q_omega.append(q_trace)
                self.s_omega.append(score)

    def compute_q_trace_and_score_of_trace(self, trace_id):
        """
        Return the QTrace (trace of Q value) and the Score assigned to the trace.
        :param trace_id: identifier of the trace
        :return: QTrace and Score
        """
        trace = self.omega[trace_id]
        q_trace = np.apply_along_axis(func1d=self.q_eval_state, axis=0, arr=trace)
        return np.expand_dims(q_trace, axis=0), np.max(q_trace)     # q_trace of shape (1,|T|) for consistency with other structs

    def q_eval_state(self, state):
        """
        Evaluation of the current state using QNetwork.
        :param state: state to be evaluated
        :return: Q value
        """
        state_ext = state.reshape((1, 1, 25, 1))   # Compatibility with model input
        return self.agent.model.predict(state_ext)[0, 0]    # return only the value, no nasted struct

    def choose_next_level(self):
        """
        This method implement the logic in the choice of the next level.
        :return: a tuple (result, level) where
                    `result` indicates if the next level has been found, and
                    `level` is the new level found (or the current one if such level doesn't exist)
        """
        np_s_omega = np.array(self.s_omega[:])
        filter_s_omega = np_s_omega[np.nonzero(np_s_omega >= self.level)]   #TODO: CONSIDER ZENO EFFECT
        if filter_s_omega.size < self.k_best_level:    # if not found K promising traces
            return False, self.level    # not found next level, current level
        index = np.argsort(filter_s_omega)[-self.k_best_level]   # take exactly the k-th higher score
        return True, filter_s_omega[index]  # found next level, value next level

    def split_traces(self, level):
        """
        Return a boolean array according to the score of each trace compared to the given `level`
        :param level: reference level
        :return: bool array
        """
        return self.s_omega >= level    # Minimizing the score, than keep the good (lower) ones

    def compute_prefix_of_trace(self, trace_id, level):
        """
        Return the prefix of the trace `trace_id` until the first state which has a level >= `level`
        :param trace_id: identifier of the trace
        :param level: minimum level to cut
        :return: the subtrace (prefix) until the given level
        """
        break_point = np.argwhere(self.q_omega[trace_id] >= level)[0][1]  # first state with score < level (MINIMIZING)
        return self.omega[trace_id][:, 0:break_point+1]    # return sub trace up to the breakpoint (included)

    def falsification_procedure(self, trace_id):
        """
        Ending procedure in which the probability of falsification is computed, according to the
        :param trace_id:
        :return:
        """
        print("[Info] FALSIFICATION!")
        self.falsification_counter = self.falsification_counter + 1
        import ipdb
        ipdb.set_trace()
        self.evaluate_all_traces_with_qlearning()  # for each trace, compute q values and score (min qval in trace)
        self.level = self.s_omega[trace_id]     #last level is the score of the current trace
        bool_split = self.split_traces(self.level)  # check each trace if is above or below the current level
        k_prom_traces = 0
        k_non_prom_traces = 0
        for i, b in enumerate(bool_split):
            if b == True:
                k_prom_traces = k_prom_traces + 1
            else:
                k_non_prom_traces = k_non_prom_traces + 1
        self.level_counter = self.level_counter + 1
        # Store level info
        self.level_list.append(self.level)
        self.level_prob_list.append(k_prom_traces / (k_prom_traces + k_non_prom_traces))  # Compute prob as avg
        error_prob = np.prod(self.level_prob_list)
        self.error_prob_list.append(error_prob)  # The error prob is given by the prod. of cond.prob.

        # Plot and print
        self.plot_episode()
        self.plot_score(highlight_last=True, error_trace_id=trace_id)
        print("[Info] Levels: " + str(self.level_list))
        print("[Info] Cond. Prob: " + str(self.level_prob_list))
        print("[Info] Error Prob: " + str(error_prob))

        if self.step_by_step_exec:
            input()

    def plot_episode(self, save_fig=True):
        """
        Plot the current episode (trajectory).
        :param save_fig: boolean for save the figure on disk
        :return: -
        """
        OUT_DIR = os.path.join("out", "fig")
        fig = plt.figure(1)
        self.env.render()
        if save_fig:
            fig_name = strftime("falsification_%Y-%m-%d_%H-%M-%S", gmtime())
            plt.savefig(os.path.join(OUT_DIR, fig_name))

    def plot_score(self, highlight_last=False, error_trace_id=None):
        """
        Plot the score distribution and the current level.
        :param highlight_last: mark the last level with a different color
        :return: -
        """
        clear_flag = False
        if self.level_counter == 1:
            clear_flag = True
        fig = plt.figure(2, clear=clear_flag)
        x = np.linspace(0, self.num_particles-1, self.num_particles)    #index from 0 for consistency w.r.t. the algorithm
        y = self.s_omega
        plt.xlim((0, self.num_particles+1))
        plt.xlabel("Particles")
        plt.ylabel("Trace Score")
        plt.stem(x, y, markerfmt='.')
        for i, level in enumerate(self.level_list):
            if highlight_last and i == len(self.level_list)-1:
                plt.axhline(y=level, color='g', linestyle='-')
                if error_trace_id is not None:
                    plt.scatter(error_trace_id, y[error_trace_id], marker='*', s=2)  #mark trace with error
            else:
                plt.axhline(y=level, color='r', linestyle='--')
        plt.pause(0.001)