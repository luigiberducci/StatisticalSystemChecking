import collections
import math
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import os
from time import gmtime, strftime
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError as MSE

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class RLISAgent:
    """
    Implementation of RLIS (RL+ImpSplit) agent
    for Statistical System Checking based on Deep Reinforcement Learning.
    """
    def __init__(self, model_manager, model, memory, mem_warmup_steps, opt="sgd", lr=0.01, opt_params=[], loss_name="mse", level_dir="out", trace_dir="out", model_dir="out"):
        self.trace_dir = trace_dir
        self.level_dir = level_dir
        self.model_dir = model_dir
        self.mem_warmup_steps = mem_warmup_steps
        self.opt_name = opt
        self.lr = lr
        self.opt_params = opt_params
        self.loss_name = loss_name
        self.replay_memory = memory
        self.model_manager = model_manager
        self.model = model
        self.model.compile(optimizer=self.optimizer(), loss=[self.loss()])
        self.template_ed_phase_start = "[Info] Start Mean Robustness estimation. Eps: {}, Delta: {}, N: {}"
        self.template_ed_phase_end = "[Info] End Mean Robustness estimation. Mean Min Rob: {}"
        self.template_episode = " {}/{}: episode: {}, level: {}, particle: {}, episode steps: {}, episode reward: {:.4f}, episode rob: {:.4f}, new memory entries: {}"
        self.template_new_level = "[Info] {}-th LEVEL FOUND: {}, Considering {}/{} traces"
        self.template_not_found_level = "[Info] Not found new level. Last level: {}, Num levels: {}"
        self.template_inc_parts = "[Info] Increment Number of particles. N = {}"
        self.template_epoch = " [Info] epoch {}: init Q value {}, mean train loss {}, train accuracy {}"

    def print_config(self):
        print("[Info] Agent (RLIS) Configuration")
        print("[Info] Optimizer: {}".format("SGD"))
        print("[Info] Loss: {}".format("MSE"))
        print("[Info] Learning rate: {}".format(self.lr))
        print("[Info] Replay memory: {}".format(self.replay_memory.get_config()))
        print()

    def optimizer(self):
        if self.opt_name == "sgd":
            return SGD(lr=self.lr)
        if self.opt_name == "sgd0":
            return SGD(lr=self.lr, decay=0)
        elif self.opt_name == "adam":
            return Adam(lr=self.lr)
        return self.opt_name

    def loss(self):
        return self.loss_name

    def save_weights(self, step_counter=0, print_info=True):
        info_template = "[Info] Saved weights: step: {}, file: {}"
        weights_filename = os.path.join(self.model_dir, "weights_{}.h5".format(step_counter))
        #self.model.save_weights(weights_filename)     # seems there is a bug in keras save_weights
        self.model.save(weights_filename)
        if print_info:
            print(info_template.format(step_counter, weights_filename))

    def test(self, sys, render=True, num_particles=100, k_particles=10, delta=0, rscale_flag=True, scale_factor=1):
        # Exploratory phase (estimation of mean rob for Robustness scaling)
        eps, delt = 0.1, 0.01   # (e,d)-approx of mean rob
        if rscale_flag:
            if scale_factor == 1:
                scale_factor = self.run_ed_rob_estimation(sys, eps, delt)
            sys.set_rob_scaling(scale_factor)
            print(self.template_ed_phase_end.format(scale_factor))
        prob, num_falsifications, _, step_counter, _ = self.run_importance_splitting(sys, num_particles, k_particles, delta,
                                                                                    render, learning=False)
        return prob, num_falsifications, step_counter

    def train(self, sys, max_sim_steps=10000, render=False, max_num_particles=500, min_num_particles=100, particles_inc = 50,
              k_particles=10, exploratory_steps=10000, delta=0, weight_interval=np.Inf, rscale_flag=True):
        step_counter = 0
        episode_counter = 0
        falsification_counter = 0
        prob_list = []
        first_fals_occurrence = np.Inf
        num_particles = min_num_particles
        # Exploratory phase (estimation of mean rob for Robustness scaling)
        # Note: since there is NO learning, I don't count these steps
        eps, delt = 0.1, 0.01   # (e,d)-approx of mean rob
        if rscale_flag:
            avg_min_rob = self.run_ed_rob_estimation(sys, eps, delt)
            sys.set_rob_scaling(avg_min_rob)    # set scaling parameter, dividing by 2 means scale in [0,2]
            print(self.template_ed_phase_end.format(avg_min_rob))
        # Training loop
        print("[Info] Train (ISplit) Configuration")
        print("[Info] Num steps: {}".format(max_sim_steps))
        print("[Info] N: {}, K: {}, Delta: {}".format(num_particles, k_particles, delta))
        print()
        while step_counter < max_sim_steps:
            # if step_counter < 500000:
            #     render = False
            # else:
            #     render = True
            error_prob, num_fals, cur_first_fals_occurrence, step_counter, episode_counter = self.run_importance_splitting(sys, num_particles, k_particles, delta,
                                                                                         render, learning=True, step_counter=step_counter,
                                                                                         episode_counter=episode_counter, max_sim_step=max_sim_steps,
                                                                                         save_weights_interval=weight_interval)
            if num_fals>0:
                first_fals_occurrence = cur_first_fals_occurrence if falsification_counter == 0 else first_fals_occurrence
                prob_list.append(error_prob)
                falsification_counter = falsification_counter + num_fals
            else:
                # run exploratory phase once
                expl_epsilon = 0.5
                expl_num_part = 20
                exploratory_steps = 0   #disable exploration
                num_fals, cur_first_fals_occurrence, step_counter, episode_counter = self.run_exploratory_phase(sys, expl_num_part,
                                                                                                                exploratory_steps, expl_epsilon,
                                                                                                                step_counter, episode_counter,
                                                                                                                max_sim_steps, weight_interval,
                                                                                                                render=render)
                # Update num particles adaptively to avoid undersampling
                num_particles = min(max_num_particles, num_particles + particles_inc)
                #print(self.template_inc_parts.format(num_particles))
        # Save model weights at the end
        self.save_weights(step_counter)
        return prob_list, falsification_counter, first_fals_occurrence

    def run_ed_rob_estimation(self, sys, e=0.1, d=0.1):
        N = math.ceil(1 / (e ** 2) * math.log(1 / d))  # from Bernstein's inequality
        sum_rob = 0
        print(self.template_ed_phase_start.format(e, d, N))
        for i in range(N):
            sys.reset_init_state()
            sys.run_system()
            _, rob = sys.get_min_robustness()
            sum_rob += rob  # cumulative sum
        mean = sum_rob / N
        return mean

    def run_exploratory_phase(self, sys, num_particles, max_expl_step, expl_epsilon, step_counter=0, episode_counter=0,
                              max_sim_step=np.Inf, save_weights_interval=np.Inf, render=False, learning=True,):
        omega = list()  # List of trace of states
        prefix_list = [sys.original_s0] * num_particles # List of prefix from which start the next level, init only `s0`
        trace_ids_falsification = []  # List of indices of traces which exhibited error
        next_breakpoint_weights = save_weights_interval     # when step>next_bp_w then save the model weights
        falsification_counter, first_fals_step = 0, np.Inf
        while max_expl_step > 0:
            # Execution and Learning phases
            for id in range(num_particles):
                if max_expl_step <= 0:
                    break
                # Run system
                episode_counter = episode_counter + 1
                sampled_prefix = prefix_list[id]
                trace, reward, reward_arr, i_init_run, i_max_reward, error_found = self.run_system_from_prefix(sys, sampled_prefix)
                # Store trace
                assert id <= len(omega)  # at most is the new element to insert
                if len(omega) > id:
                    omega[id] = trace
                else:
                    omega.append(trace)  # store the current trace (episode)

                # Update steps and save model weights
                episode_steps = trace.shape[1] - sys.last_init_state - 1
                max_expl_step -= episode_steps
                step_counter = step_counter + episode_steps  # update step counter
                if render:
                    log_info = "EXPL: Episode {}, Steps {} - Reward {}".format(episode_counter, step_counter, reward)
                    sys.render(title=log_info)
                if step_counter > next_breakpoint_weights:
                    self.save_weights(step_counter)
                    next_breakpoint_weights = step_counter + save_weights_interval

                # Manage eventually falsification
                if error_found:
                    falsification_counter = falsification_counter + 1
                    first_fals_step = step_counter if falsification_counter == 1 else first_fals_step
                    trace_ids_falsification.append(id)
                    log_info = "Exploration - Episode {}, Steps {} - Reward {}".format(episode_counter, step_counter, reward)
                    print("[Info] FALSIFICATION! {}".format(log_info))
                    name_prefix = "expl_falsification_{}_{}".format(episode_counter, falsification_counter)
                    sys.render(title=log_info, save_fig=True, prefix=name_prefix, out_dir=self.trace_dir)

                # exploratory - Learning phase (if enabled)
                if learning:
                    import ipdb
                    ipdb.set_trace()
                    for i in range(reward_arr.shape[0]):  # reward for all states in the trace
                        terminal = (i == trace.shape[1] - 1)  # never used
                        observation = trace[:, i]
                        self.replay_memory.append(observation, None, reward_arr[i], terminal)  # action is None
                        if step_counter >= self.mem_warmup_steps:  # Learning step
                            experience = self.replay_memory.sample(self.model_manager.batch_size)
                            batch_state = np.array([exp.state0[0][0:self.model.state_variables] for exp in experience])
                            batch_reward = np.array([exp.reward for exp in experience])
                            self.model.train_on_batch(batch_state, batch_reward)
                # Log info
                text = self.template_episode.format(step_counter, max_sim_step, episode_counter, "expl", id,
                                                    episode_steps, reward, sys.robustness,
                                                    i_max_reward - i_init_run + 1)
                print(text)
            if step_counter >= max_sim_step: # end procedure
                # no message, simply end
                break

            # Evaluation phase
            q_omega, s_omega = self.evaluate_all_traces_with_qlearning(omega)
            # Update prefixes for next iteration
            prefix_list = self.epsilon_greedy_update_of_prefix_list(omega, q_omega, epsilon=expl_epsilon)
        return falsification_counter, first_fals_step, step_counter, episode_counter

    def epsilon_greedy_update_of_prefix_list(self, omega, q_omega, epsilon=0.5):
        num_particles = len(omega)
        best_breakpoints = [np.argmax(q_trace) for q_trace in q_omega]   # find the best breakpoint for each trace (max q-value)
        rand_breakpoints = [np.random.randint(0, trace.shape[1]-1) for trace in omega]
        random_choice = np.random.rand(num_particles) <= epsilon    # with prob "epsilon" take a rnd cut
        final_breakpoints = [rand_breakpoints[i] if random_choice[i] else best_breakpoints[i] for i in range(num_particles)]
        prefix_list = [omega[i][:, 0:final_breakpoints[i] + 1] for i in range(num_particles)]
        return prefix_list

    def run_importance_splitting(self, sys, num_particles, k_particles, delta_level, render, learning,
                                 step_counter=0, episode_counter=0, max_sim_step=np.Inf, save_weights_interval=np.Inf):
        omega = list()          # List of trace of states
        prefix_list = [sys.original_s0]    # List of prefix from which start the next level, init only `s0`
        available_indices = collections.deque(maxlen=num_particles)
        for i in range(num_particles):  # init all indices available
            available_indices.append(i)
        level = np.NINF  # Current level, since minimization initialize as INF
        level_list = []         # List of level thresholds
        level_prob_list = []    # List of conditional probabilities
        error_prob = 0          # Error probability estimation
        level_counter = 0       # Number of levels found
        trace_ids_falsification = []    # List of indices of traces which exhibited error
        falsification_counter = 0       # Counter of number of falsifications
        first_fals_step = np.Inf
        reward_list = []        # History of reward (maybe useless)
        # in this way, it will save the weights at each first iteration of importance splitting
        next_breakpoint_weights = self.mem_warmup_steps   # save at the begin, then when step>next_bp_w then save the model weights

        # Importance Splitting
        while True:
            # Execution and Learning phases
            while available_indices and step_counter<max_sim_step:
                # Run system
                episode_counter = episode_counter + 1
                sampled_prefix = rand.sample(prefix_list, 1)[0]  # sample return a list of 1 (k) element (avoid mem problem)
                trace, reward, reward_arr, i_init_run, i_max_reward, error_found = self.run_system_from_prefix(sys, sampled_prefix)
                reward_list.append(reward)
                # Store trace
                id = available_indices.popleft()  # get the first available index in omega
                assert id <= len(omega)  # at most is the new element to insert
                if len(omega) > id:
                    omega[id] = trace
                else:
                    omega.append(trace)  # store the current trace (episode)

                # Update steps and save model weights
                episode_steps = trace.shape[1] - sys.last_init_state - 1
                step_counter = step_counter + episode_steps  # update step counter
                if render:
                    log_info = "Episode {}, Steps {} - Reward {}".format(episode_counter, step_counter, reward)
                    sys.render(title=log_info)

                # Manage eventually falsification
                if error_found:
                    falsification_counter = falsification_counter + 1
                    first_fals_step = step_counter if falsification_counter == 1 else first_fals_step
                    trace_ids_falsification.append(id)
                    log_info = "Episode {}, Steps {} - Reward {}".format(episode_counter, step_counter, reward)
                    print("[Info] FALSIFICATION! {}".format(log_info))
                    name_prefix = "falsification_{}_{}".format(episode_counter, falsification_counter)
                    # save figure only during training
                    sys.render(title=log_info, save_fig=learning, prefix=name_prefix, out_dir=self.trace_dir)

                # Learning phase (if enabled)
                if learning:
                    for i in range(reward_arr.shape[0]):  # reward for all states
                        terminal = (i == trace.shape[1] - 1)    # note: never used
                        observation = trace[self.model_manager.state_filter, i]   # state i filtered (only meaningful variables)
                        reward_i = reward_arr[i]    # reward of state i
                        self.replay_memory.append(observation, None, reward_i, terminal)  # action is None
                        if step_counter >= self.mem_warmup_steps:  # Learning step
                            experience = self.replay_memory.sample(self.model_manager.batch_size)
                            batch_state = np.array([exp.state0[0] for exp in experience])
                            batch_reward = np.array([exp.reward for exp in experience])
                            self.model.train_on_batch(batch_state, batch_reward)
                    # Save model weights (only during training)
                    if step_counter > next_breakpoint_weights:
                        self.save_weights(step_counter)
                        next_breakpoint_weights = step_counter + save_weights_interval
                # Log info
                text = self.template_episode.format(step_counter, max_sim_step, episode_counter, level_counter, id,
                                                    episode_steps, reward, sys.robustness, reward_arr.shape[0])
                print(text)

            if falsification_counter > 0:   # Error found
                error_prob = self.falsification_procedure(level_list, level_prob_list, omega, trace_ids_falsification, learning)
                break
            elif step_counter >= max_sim_step: # end procedure
                # no message, simply end
                break

            # Evaluation phase
            q_omega, s_omega = self.evaluate_all_traces_with_qlearning(omega, step_counter=step_counter)
            found_next_level, next_level = self.choose_next_level(s_omega, level, delta_level, k_particles)
            if not found_next_level:
                print(self.template_not_found_level.format(level, level_counter))
                break

            # Debug - Print Qvalues
            if render:
                plt.figure(7, clear=True)
                for q_trace in q_omega:  # iterate over all particles
                    plt.plot(q_trace[0])
                plt.hlines(next_level, xmin=0, xmax=80)
                plt.pause(0.001)

            # Found next level, update level info
            level = next_level
            level_counter = level_counter + 1
            level_list.append(level)
            # Split traces and compute conditional probability
            bool_split = s_omega >= level  # which particles reached the level
            num_prom_traces = np.count_nonzero(s_omega >= level)    # count particles which reached the level
            # Update prefixes for next iteration
            prefix_list, available_indices = self.update_prefix_list_with_new_level_prefixes(omega, q_omega, bool_split, level)
            # Append new level probability
            level_prob_list.append(num_prom_traces / num_particles)  # Compute prob as avg
            print("[Info] Len prefix list: {}".format(len(prefix_list)))
            if len(prefix_list) == 0:   # no breakpoints, probably qvalues>=level only in last states
                print("[Info] Prefix list empty. Reset.".format(len(prefix_list)))
                break
            # Log info
            print(self.template_new_level.format(level_counter, level, num_prom_traces, num_particles))
            #self.plot_score(s_omega, level_list, trace_ids_falsification)
        return error_prob, falsification_counter, first_fals_step, step_counter, episode_counter

    def run_system_from_prefix(self, sys, prefix):
        if prefix is None:
            sys.reset_init_state()
        else:
            prefix = prefix.copy()
            sys.set_prefix(prefix)
        sys.run_system()
        return sys.get_trace(), sys.reward, sys.reward_array, sys.last_init_state, sys.i_max_reward, sys.is_current_trace_false()

    def falsification_procedure(self, level_list, level_prob_list, omega, trace_ids_falsification, learning=True):
        """
        Ending procedure in which the probability of falsification is computed, according to the cond. prob. of levels.
        :return: -
        """
        level_counter = len(level_list)
        num_particles =  len(omega)
        _, s_omega = self.evaluate_all_traces_with_qlearning(omega)  # for each trace, compute q values and score (min qval in trace)
        # Compute Falsification level (minimum value among trace which exhibited falsification)
        fals_trace_id = np.argmin(np.array([s_omega[i] for i in trace_ids_falsification]))
        level = s_omega[trace_ids_falsification[fals_trace_id]]
        # Split on last level BASED ON FALSIFICATION (not score)
        k_falsifications = len(trace_ids_falsification)
        level_counter = level_counter + 1
        # Store level info
        level_list.append(level)
        level_prob_list.append(k_falsifications / (num_particles))  # Compute prob as avg
        error_prob = np.prod(level_prob_list)
        # Plot and print
        self.plot_score(s_omega, level_list, trace_ids_falsification, clear_flag=True, highlight_last=True, save_fig=learning)
        print("[Info] Complete ISplit Iteration")
        print("[Info] Levels: " + str(level_list))
        print("[Info] Cond. Prob: " + str(level_prob_list))
        print("[Info] Variance Probs: " + str(np.var(level_prob_list)))
        print("[Info] Error Prob: " + str(error_prob))
        print("")
        return error_prob

    def evaluate_all_traces_with_qlearning(self, omega, step_counter=0):
        """
            Evaluation of the current set of traces (particles) using Q Network.
            This method fills `q_omega` and `s_omega`.
            :return: -
        """
        q_omega = list()
        s_omega = list()

        for i in range(len(omega)):     #iterate over all particles
            q_trace, score = self.compute_q_trace_and_score_of_trace(omega[i])  # qtrace of qval for each state, score computed by them
            assert i <= len(q_omega)    # at most is the new element to insert
            if len(q_omega) > i:
                q_omega[i] = q_trace
                s_omega[i] = score
            else:
                q_omega.append(q_trace)
                s_omega.append(score)
        return q_omega, s_omega

    def compute_q_trace_and_score_of_trace(self, trace):
        """
        Return the QTrace (trace of Q value) and the Score assigned to the trace.
        :param trace_id: identifier of the trace
        :return: QTrace and Score
        """
        trace = tf.data.Dataset.from_tensor_slices(np.swapaxes(trace[self.model_manager.state_filter], 0, 1).astype(np.float32)).batch(self.model_manager.batch_size)
        for i, batch_state in enumerate(trace):
            batch_state = batch_state.numpy()
            if i == 0:
                q_trace = self.model(batch_state).numpy()
            else:
                q_trace = np.vstack([q_trace, self.model(batch_state).numpy()])
        # error detection
        if np.isnan(q_trace[0]):
            print("[Error] Found NaN. Consider to decrease the learning rate because of too big gradient.")
        #q_trace = q_trace - q_trace[0]     #variant score w.r.t baseline
        return np.swapaxes(q_trace, 0, 1), np.max(q_trace)  # q_trace of shape (1,|T|) for consistency with other structs

    def choose_next_level(self, s_omega, last_level, delta_level, k_best_level):
        """
        This method implement the logic in the choice of the next level.
        :return: a tuple (result, level) where
                    `result` indicates if the next level has been found, and
                    `level` is the new level found (or the current one if such level doesn't exist)
        """
        np_s_omega = np.array(s_omega[:])
        if any(np.isnan(np_s_omega)):
            print("[Error] Found NaN in S-trace. Why? Learning rate too high?")
        filter_s_omega = np_s_omega[np.nonzero(np_s_omega > last_level + delta_level)]  # Delta>0 solves Zeno Effect
        # Note: if delta>0, go to the next level even if less than K
        if filter_s_omega.size == 0 or (filter_s_omega.size < k_best_level and delta_level == 0):  # if not found K promising traces
            return False, last_level  # not found next level, current level
        k_best_level = min(k_best_level, len(filter_s_omega))   # to address the case in which delta>0 and less than K promising traces
        index = np.argsort(filter_s_omega)[-k_best_level]  # take exactly the k-th higher score
        next_level = filter_s_omega[index]
        return True, next_level  # found next level, value next level

    def update_prefix_list_with_new_level_prefixes(self, omega, q_omega, bool_split, level):
        num_particles = len(bool_split)
        prefix_list = list()
        available_indices = collections.deque(maxlen=num_particles)
        #import ipdb
        #ipdb.set_trace()
        for i, b in enumerate(bool_split):
            if b:
                # Promising trace, compute prefix and add it to env as restarting point
                # note: consider breakpoint all states but the last, otherwise we simulate for 0 steps (starvation)
                for break_point in np.argwhere(q_omega[i][0][:-1] >= level).flatten():  # all states with score >= level
                    prefix = omega[i][:, 0:break_point + 1]
                    prefix_list.append(prefix)
                available_indices.append(i)  # trace `i` will be overwritten
            else:
                # Non-promising trace, flag it for next replacement
                available_indices.append(i)  # trace `i` will be overwritten
        return prefix_list, available_indices

    def plot_score(self, s_omega, level_list, trace_ids_falsification=[], clear_flag=None, highlight_last=False, save_fig=True):
        """
        Plot the score distribution and the current level.
        :param highlight_last: mark the last level with a different color
        :return: -
        """
        clear_flag = len(level_list) == 1 if clear_flag is None else clear_flag
        num_particles = len(s_omega)
        level_counter = len(level_list)

        fig = plt.figure(2, clear=clear_flag)
        x_axis_len = min(num_particles, len(s_omega))
        x = np.linspace(0, x_axis_len - 1, x_axis_len)  # index from 0 for consistency w.r.t. the algorithm
        y = s_omega
        plt.xlim((0, num_particles + 1))
        # plt.ylim((0.5, 1.0))
        plt.xlabel("Particles")
        plt.ylabel("Trace Score")
        plt.stem(x, y, markerfmt='.')
        for i, level in enumerate(level_list):
            for error_trace_id in trace_ids_falsification:
                if error_trace_id == -1:  # falsification in warmup episodes (not stored trace)
                    continue
                plt.scatter(error_trace_id, y[error_trace_id], marker='*', c='g', s=80)  # mark trace with error
            if highlight_last and i == len(level_list) - 1:
                plt.axhline(y=level, color='g', linestyle='-')
            else:
                plt.axhline(y=level, color='r', linestyle='--')
        plt.pause(0.001)
        if highlight_last and save_fig:  # save only last plot
            fig_name = strftime("level_splitting_%Y-%m-%d_%H-%M-%S", gmtime())
            plt.savefig(os.path.join(self.level_dir, fig_name))







