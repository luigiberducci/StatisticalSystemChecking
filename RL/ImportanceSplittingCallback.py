from keras.callbacks import Callback
import collections
import os
from time import gmtime, strftime
import numpy as np
import matplotlib.pyplot as plt

class ImportanceSplittingCallback(Callback):
    def __init__(self, env, agent, num_particles=100, k=10, warmup_skip=50, delta_level=0, run_ispl=True, outdir='out'):
        self.env = env  #Store the environment for prefix interaction
        self.agent = agent  #Store the agent for later using of the qnet
        self.warmup_episodes = warmup_skip  #iterations to skip to allow the qvalue to have sense in the first states
        self.use_i_splitting = run_ispl  #Flag to run isplit or simple monte carlo simulations
        self.outdir = outdir #output directory for logging
        self.level_outdir = os.path.join(outdir, "levels")
        self.traces_outdir = os.path.join(outdir, "traces")
        if not os.path.exists(self.level_outdir):   #Create if necessary
            os.makedirs(self.level_outdir, exist_ok=True)
        if not os.path.exists(self.traces_outdir):
            os.makedirs(self.traces_outdir, exist_ok=True)
        # Debug options
        self.falsification_counter = 0  #no reset
        self.step_by_step_exec = False  # enable input at each new level, reset or falsification
        # Configure ISplit
        adaptive_level_splitting = True #ALWAYS
        fixed_levels = []               #STATIC LEVELS DEPRECATED
        assert adaptive_level_splitting is True or len(fixed_levels) > 0    #fixed_level -> not adaptive
        assert delta_level == 0 or adaptive_level_splitting is True         #delta -> adaptive
        assert k <= 0 or adaptive_level_splitting==True                     #k>0 -> adaptive
        assert num_particles > 0                                            #trivial particles>0
        self.num_particles = num_particles  # Number of particles for ISplit
        self.k_best_level = k  # Number of particles for next level computation
        self.adaptive_level_splitting = adaptive_level_splitting
        self.delta_level = delta_level
        self.fixed_level = fixed_levels
        self.omega = list()     # List of trace of states
        self.q_omega = list()   # List of trace of QValue of each state
        self.s_omega = list()   # List of score for each trace
        self.level = np.NINF     # Current level, since minimization initialize as INF
        self.level_counter = 0    #current number of levels
        self.exist_falsification = False    # flag to mark the last iteration of ISplit, when Fals. occurred
        self.trace_ids_falsification = []
        # Storing level info
        self.level_list = []        # list of level in current execution (each new restart, reset)
        self.level_prob_list = []   # list of level reach. prob in current exec (each restart, reset)
        self.error_prob_list = []   # list of falsification probability of all execution (permanent)
        # Initialize indices for writing in `omega`
        self.available_indices = collections.deque(maxlen=self.num_particles)
        for i in range(self.num_particles):     #init: all indices available
            self.available_indices.append(i)
        #Info printing
        self.print_info_config()

    def print_info_config(self):
        print("[Info] ISplitting Configuration")
        print("[Info] Use ISplitting: {}".format(self.use_i_splitting))
        print("[Info] Output Dirs: {}, {}, {}".format(self.outdir, self.level_outdir, self.traces_outdir))
        print("[Info] Num particles: {}".format(self.num_particles))
        print("[Info] Adaptive Multilevel Splitting: {}, delta={}, k={}".format(self.adaptive_level_splitting, self.delta_level, self.k_best_level))
        print("[Info] Fixed Level Splitting: {}".format(str(self.fixed_level)))

    def reset_importance_splitting(self):
        # Reset env prefix from the original one
        self.env.reset_original_prefix_list()
        # All data structures empty
        self.omega = list()  # List of trace of states
        self.q_omega = list()  # List of trace of QValue of each state
        self.s_omega = list()  # List of score for each trace
        self.level = np.NINF  # Current level, since minimization initialize as INF
        self.level_counter = 0    #current number of levels
        self.exist_falsification = False    # flag to mark the last iteration of ISplit, when Fals. occurred
        self.trace_ids_falsification = []
        # Rendering of levels
        self.level_list = []
        self.level_prob_list = []   # list of level reach. prob in current exec (each restart, reset)
        # Initialize indices for writing in `omega`
        self.available_indices.clear()
        for i in range(self.num_particles):  # init: all indices available
            self.available_indices.append(i)

    def on_episode_end(self, episode, logs={}):
        #self.plot_episode()
        if not self.use_i_splitting:        #disable ispl, run only MC+Qlearning (no exploration policy)
            return
        if episode < self.warmup_episodes:
            return

        if self.available_indices:              # Not empty
            last_i = self.store_current_trace()
            if self.env.is_current_trace_false():   # Error found
                print("[Info] FALSIFICATION!")
                self.falsification_counter = self.falsification_counter + 1
                self.exist_falsification = True
                self.trace_ids_falsification.append(last_i)
                self.plot_episode()
        if not self.available_indices:  # Empty = We collected all the particles required
            # Note: don't use `else` otherwise we loose an iteration
            if self.exist_falsification:
                self.falsification_procedure()  # Plot, Compute probability, ...
                self.reset_importance_splitting()     # Reset ISplit from scratch
            else:
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
        assert not self.exist_falsification
        self.evaluate_all_traces_with_qlearning()   #for each trace, compute q values and score (min qval in trace)
        found_next_level, next_level = self.choose_next_level()
        if not found_next_level:        # No next level value (max num reached for static, no better level for adaptive)
            self.no_new_level_then_reset()
            return
        bool_split, k_prom_traces, k_non_prom_traces = self.split_traces(next_level)  # check each trace if is above or below the current level
        if k_prom_traces == 0:
            self.no_new_level_then_reset()
            return
        else:   #at least a good trace has been found
            self.level = next_level
            self.update_prefix_list_with_new_level_prefixes(bool_split)
            # Store level info
            self.level_list.append(self.level)
            self.level_prob_list.append(k_prom_traces/(k_prom_traces+k_non_prom_traces))    #Compute prob as avg
            self.level_counter = self.level_counter + 1
            print("[Info] {}-TH LEVEL FOUND: {}, Considering {}/{} traces".format(self.level_counter, self.level, k_prom_traces, self.num_particles))
            self.plot_score()
            if self.step_by_step_exec:
                input()

    def update_prefix_list_with_new_level_prefixes(self, bool_split):
        self.env.clear_prefix_list()  # once computed the new level, reset the prefix list to empty
        for i, b in enumerate(bool_split):
            if b == True:
                # Promising trace, compute prefix and add it to env as restarting point
                prefix = self.compute_prefix_of_trace(i, self.level)
                self.env.add_prefix(prefix)
            else:
                # Non-promising trace, flag it for next replacement
                self.available_indices.append(i)  # trace `i` will be overwritten

    def no_new_level_then_reset(self):
        print("[Info] NOT FOUND NEW LEVEL, Current Best Level is {}".format(self.level))
        print(self.level_counter)
        if self.step_by_step_exec:
            input()
        self.reset_importance_splitting()  # Restart ISplit from scratch

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
        num_observed_variables = self.env.observation_space.shape[0]
        state_ext = state.reshape((1, 1, num_observed_variables, 1))  # ASSUME FLATTEN OBSERVATION (25 for ekf, 2 for succruns)
        q_val = self.agent.model.predict(state_ext)[0, 0]    # return only the value, no nasted struct
        return q_val

    def choose_next_level(self):
        """
        This method implement the logic in the choice of the next level.
        :return: a tuple (result, level) where
                    `result` indicates if the next level has been found, and
                    `level` is the new level found (or the current one if such level doesn't exist)
        """
        assert not self.exist_falsification
        if self.adaptive_level_splitting is True:
            np_s_omega = np.array(self.s_omega[:])
            # TODO IF >LEVEL+DELTA, ALSO <K PARTICLES
            filter_s_omega = np_s_omega[np.nonzero(np_s_omega >= self.level + self.delta_level)]  # Delta>0 solves Zeno Effect
            if filter_s_omega.size < self.k_best_level:  # if not found K promising traces
                return False, self.level  # not found next level, current level
            index = np.argsort(filter_s_omega)[-self.k_best_level]  # take exactly the k-th higher score
            next_level = filter_s_omega[index]
        else:
            number_of_levels = 5
            FIXED_LEVEL = np.array([0.01, 0.025, 0.05, 0.1, 0.25])
            if self.level_counter >= len(FIXED_LEVEL)-1:
                return False, self.level  # reached max number of levels
            next_level = FIXED_LEVEL[self.level_counter]    #next available level
        return True, next_level  # found next level, value next level

    def split_traces(self, level):
        """
        Return a boolean array according to the score of each trace compared to the given `level`
        :param level: reference level
        :return: bool array, number of true, number of false
        """
        bool_split = self.s_omega >= level    # Minimizing the score, than keep the good (lower) ones
        k_prom_traces = np.count_nonzero(self.s_omega >= level)
        k_non_prom_traces = self.num_particles - k_prom_traces
        return bool_split, k_prom_traces, k_non_prom_traces

    def compute_prefix_of_trace(self, trace_id, level):
        """
        Return the prefix of the trace `trace_id` until the first state which has a level >= `level`
        :param trace_id: identifier of the trace
        :param level: minimum level to cut
        :return: the subtrace (prefix) until the given level
        """
        break_point = np.argwhere(self.q_omega[trace_id] >= level)[0][1]  # first state with score < level (MINIMIZING)
        return self.omega[trace_id][:, 0:break_point+1]    # return sub trace up to the breakpoint (included)

    def falsification_procedure(self, plot_flag=True):
        """
        Ending procedure in which the probability of falsification is computed, according to the cond. prob. of levels.
        :return: -
        """
        self.evaluate_all_traces_with_qlearning()  # for each trace, compute q values and score (min qval in trace)
        fals_trace_id, self.level = self.compute_falsification_level() #choose next level among the traces which lead to fals.
        # Split on last level BASED ON FALSIFICATION (not score)
        k_falsifications = len(self.trace_ids_falsification)
        k_non_falsifications = self.num_particles - k_falsifications
        self.level_counter = self.level_counter + 1

        # Store level info
        self.level_list.append(self.level)
        self.level_prob_list.append(k_falsifications / (k_falsifications + k_non_falsifications))  # Compute prob as avg
        error_prob = np.prod(self.level_prob_list)
        self.error_prob_list.append(error_prob)  # The error prob is given by the prod. of cond.prob.

        # Plot and print
        if plot_flag:
            self.plot_score(highlight_last=True)
        print("[Info] Complete ISplit Iteration")
        print("[Info] Levels: " + str(self.level_list))
        print("[Info] Cond. Prob: " + str(self.level_prob_list))
        print("[Info] Error Prob: " + str(error_prob))
        print("")

        if self.step_by_step_exec:
            input()

    def compute_falsification_level(self):
        """
        Return the level in the iteration in which at least 1 Falsification occurred.
        In particular, among all falsifications occurred, return the minimum score of error trace.
        IN THIS WAY, WE ENSURES THAT w in ERR_TRACES -> score(w)>=falsification_level
        :return: id of trace with lower score (among fals. traces), level for last iteration of ISplit
        """
        min_level = np.Inf
        min_id = -1
        for trace_id in self.trace_ids_falsification:
            if self.s_omega[trace_id] < min_level:
                min_level = self.s_omega[trace_id]
                min_id = trace_id
        return min_id, min_level

    def plot_episode(self, save_fig=True):
        """
        Plot the current episode (trajectory).
        :param save_fig: boolean for save the figure on disk
        :return: -
        """
        fig = plt.figure(1)
        self.env.render()
        if save_fig:
            fig_name = strftime("falsification_%Y-%m-%d_%H-%M-%S", gmtime())
            plt.savefig(os.path.join(self.traces_outdir, fig_name))

    def plot_score(self, highlight_last=False, save_fig=True):
        """
        Plot the score distribution and the current level.
        :param highlight_last: mark the last level with a different color
        :return: -
        """
        clear_flag = False
        if self.level_counter == 1:
            clear_flag = True
        fig = plt.figure(2, clear=clear_flag)
        x_axis_len = min(self.num_particles, len(self.s_omega))
        x = np.linspace(0, x_axis_len-1, x_axis_len)    #index from 0 for consistency w.r.t. the algorithm
        y = self.s_omega
        plt.xlim((0, self.num_particles+1))
        plt.xlabel("Particles")
        plt.ylabel("Trace Score")
        plt.stem(x, y, markerfmt='.')
        for i, level in enumerate(self.level_list):
            for error_trace_id in self.trace_ids_falsification:
                plt.scatter(error_trace_id, y[error_trace_id], marker='*', c='g', s=80)  # mark trace with error
            if highlight_last and i == len(self.level_list)-1:
                plt.axhline(y=level, color='g', linestyle='-')
            else:
                plt.axhline(y=level, color='r', linestyle='--')
        plt.pause(0.001)
        if highlight_last and save_fig:     #save only last plot
            fig_name = strftime("level_splitting_%Y-%m-%d_%H-%M-%S", gmtime())
            plt.savefig(os.path.join(self.level_outdir, fig_name))
