import tensorflow as tf
import os
from rl.memory import SequentialMemory
from env.EKF import System
from model.EKFModel import EKFModel
import numpy as np
import collections
import random as rand
import matplotlib.pyplot as plt
from time import gmtime, strftime
import math

# Parameters memory and model
mem_limit = 200000
mem_window_len = 1
mem_warmup_steps = 1000
batch_size = 8

#Debug
step_by_step_exec_flag = False

#Importance Splitting
num_particles = 100
k_best_level = 10
is_warmup_episodes = 500
omega = list()     # List of trace of states
q_omega = list()   # List of trace of QValue of each state
s_omega = list()   # List of score for each trace
prefix_list = list()
available_indices = collections.deque(maxlen=num_particles)
for i in range(num_particles):  # init all indices available
    available_indices.append(i)
level = np.NINF  # Current level, since minimization initialize as INF
delta_level = 0
level_list = []
level_prob_list = []
error_prob_list = []
level_counter = 0
trace_ids_falsification = []

# Print
template_episode = " {}/{}: episode: {}, duration: -, episode steps: {}, steps per second: -, episode reward: {}, mean reward: - [-, -], mean action: 0.000 [0.000, 0.000], mean observation: - [-, -], memory update: {}"
template_epoch = " epoch {}: init q Value {}, mean train loss {}, train accuracy {}"

def process_flat_to_batch(numpy_array):
    return numpy_array.reshape((batch_size, numpy_array.shape[0]))

def store_current_trace(trace):
    i = available_indices.popleft()  # get the first available index in omega
    assert i <= len(omega)  # at most is the new element to insert
    if len(omega) > i:
        omega[i] = trace
    else:
        omega.append(trace)  # store the current trace (episode)
    return i    #return position of last writing

def evaluate_all_traces_with_qlearning():
    """
    Evaluation of the current set of traces (particles) using Q Network.
    This method fills `q_omega` and `s_omega`.
    :return: -
    """
    for i in range(num_particles):
        q_trace, score = compute_q_trace_and_score_of_trace(i)   # qtrace of qval for each state, score computed by them
        assert i <= len(q_omega)       # at most is the new element to insert
        if len(q_omega) > i:
            q_omega[i] = q_trace
            s_omega[i] = score
        else:
            q_omega.append(q_trace)
            s_omega.append(score)


def compute_q_trace_and_score_of_trace(trace_id):
    """
    Return the QTrace (trace of Q value) and the Score assigned to the trace.
    :param trace_id: identifier of the trace
    :return: QTrace and Score
    """
    trace = tf.data.Dataset.from_tensor_slices(np.swapaxes(omega[trace_id][0:9], 0, 1)).batch(batch_size)
    for i, batch_state in enumerate(trace):
        batch_state = preprocess_batch_state(batch_state.numpy())
        if i == 0:
            q_trace = model(batch_state).numpy()
        else:
            q_trace = np.vstack([q_trace, model(batch_state).numpy()])
    if np.isnan(q_trace[0]):
        print("[Info] Found NaN!")
        # import ipdb
        # ipdb.set_trace()
    if trace_id==0:
        print("[Info] Init Q Val {}".format(q_trace[0]))
    #q_trace = q_trace - q_trace[0]     #variant score w.r.t baseline
    return np.swapaxes(q_trace, 0, 1), np.max(q_trace)     # q_trace of shape (1,|T|) for consistency with other structs

def q_eval_state(state):
    """
    DEPRECATED USE DIRECTLY Q EVAL ON BATCHED TRACE
    Evaluation of the current state using QNetwork.
    :param state: state to be evaluated
    :return: Q value
    """
    batch_state = np.tile(state[0:9], (batch_size, 1))
    batch_state = preprocess_batch_state(batch_state)
    return model(batch_state).numpy()[0, 0]    # return only the value, no nasted struct

def choose_next_level():
    """
    This method implement the logic in the choice of the next level.
    :return: a tuple (result, level) where
                `result` indicates if the next level has been found, and
                `level` is the new level found (or the current one if such level doesn't exist)
    """
    assert not falsification_flag
    np_s_omega = np.array(s_omega[:])
    # TODO IF >LEVEL+DELTA, ALSO <K PARTICLES
    filter_s_omega = np_s_omega[np.nonzero(np_s_omega > level + delta_level)]  # Delta>0 solves Zeno Effect
    if filter_s_omega.size < k_best_level:  # if not found K promising traces
        return False, level  # not found next level, current level
    index = np.argsort(filter_s_omega)[-k_best_level]  # take exactly the k-th higher score
    next_level = filter_s_omega[index]
    return True, next_level  # found next level, value next level

def split_traces(level):
    """
    Return a boolean array according to the score of each trace compared to the given `level`
    :param level: reference level
    :return: bool array, number of true, number of false
    """
    bool_split = s_omega >= level    # Minimizing the score, than keep the good (lower) ones
    num_prom_traces = np.count_nonzero(s_omega >= level)
    num_non_prom_traces = num_particles - num_prom_traces
    return bool_split, num_prom_traces, num_non_prom_traces

def update_prefix_list_with_new_level_prefixes(bool_split):
    prefix_list.clear()
    for i, b in enumerate(bool_split):
        if b == True:
            # Promising trace, compute prefix and add it to env as restarting point
            prefix = compute_prefix_of_trace(i, level)
            prefix_list.append(prefix)
        else:
            # Non-promising trace, flag it for next replacement
            available_indices.append(i)  # trace `i` will be overwritten

def update_prefix_list_with_random_prefixes():
    for i in range(num_particles):
        prefix = compute_random_prefix(i)
        prefix_list.append(prefix)
        available_indices.append(i)

def compute_random_prefix(trace_id):
    break_point = rand.randint(0, 101)
    return omega[trace_id][:, 0:break_point+1]

def compute_prefix_of_trace(trace_id, level):
    """
    Return the prefix of the trace `trace_id` until the first state which has a level >= `level`
    :param trace_id: identifier of the trace
    :param level: minimum level to cut
    :return: the subtrace (prefix) until the given level
    """
    break_point = np.argwhere(q_omega[trace_id] >= level)[0][1]  # first state with score < level (MINIMIZING)
    return omega[trace_id][:, 0:break_point+1]    # return sub trace up to the breakpoint (included)

def reset_importance_splitting():
    global epoch_counter
    epoch_counter = epoch_counter + 1
    init_q_val = q_eval_state(sys.original_s0.reshape(25))
    print(template_epoch.format(epoch_counter, init_q_val, train_loss.result(), train_accuracy.result() * 100))
    # Reset env prefix from the original one
    prefix_list.clear()
    prefix_list.append(sys.original_s0)  # init prefix list with initial state
    # All data structures empty
    omega.clear()  # List of trace of states
    q_omega.clear()  # List of trace of QValue of each state
    s_omega.clear()  # List of score for each trace
    # Initialize indices for writing in `omega`
    available_indices.clear()
    for i in range(num_particles):  # init: all indices available
        available_indices.append(i)
    # reset learning metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

def falsification_procedure():
    """
    Ending procedure in which the probability of falsification is computed, according to the cond. prob. of levels.
    :return: -
    """
    global level_counter
    evaluate_all_traces_with_qlearning()  # for each trace, compute q values and score (min qval in trace)
    fals_trace_id, level = compute_falsification_level() #choose next level among the traces which lead to fals.
    # Split on last level BASED ON FALSIFICATION (not score)
    k_falsifications = len(trace_ids_falsification)
    level_counter = level_counter + 1
    # Store level info
    level_list.append(level)
    level_prob_list.append(k_falsifications / (num_particles))  # Compute prob as avg
    error_prob = np.prod(level_prob_list)
    error_prob_list.append(error_prob)  # The error prob is given by the prod. of cond.prob.
    # Plot and print
    plot_score(highlight_last=True)
    print("[Info] Complete ISplit Iteration")
    print("[Info] Levels: " + str(level_list))
    print("[Info] Cond. Prob: " + str(level_prob_list))
    print("[Info] Error Prob: " + str(error_prob))
    print("")
    if step_by_step_exec_flag:
        input()

def compute_falsification_level():
    """
    Return the level in the iteration in which at least 1 Falsification occurred.
    In particular, among all falsifications occurred, return the minimum score of error trace.
    IN THIS WAY, WE ENSURES THAT w in ERR_TRACES -> score(w)>=falsification_level
    :return: id of trace with lower score (among fals. traces), level for last iteration of ISplit
    """
    min_level = np.Inf
    min_id = -1
    for trace_id in trace_ids_falsification:
        if trace_id == -1:  #falsification during the warmup episodes (not stored trace)
            continue
        if s_omega[trace_id] < min_level:
            min_level = s_omega[trace_id]
            min_id = trace_id
    return min_id, min_level

def plot_score(highlight_last=False, save_fig=True):
    """
    Plot the score distribution and the current level.
    :param highlight_last: mark the last level with a different color
    :return: -
    """
    clear_flag = False
    if level_counter == 1:
        clear_flag = True
    fig = plt.figure(2, clear=clear_flag)
    x_axis_len = min(num_particles, len(s_omega))
    x = np.linspace(0, x_axis_len-1, x_axis_len)    #index from 0 for consistency w.r.t. the algorithm
    y = s_omega
    plt.xlim((0, num_particles+1))
    # plt.ylim((0.5, 1.0))
    plt.xlabel("Particles")
    plt.ylabel("Trace Score")
    plt.stem(x, y, markerfmt='.')
    for i, level in enumerate(level_list):
        for error_trace_id in trace_ids_falsification:
            if error_trace_id == -1:    # falsification in warmup episodes (not stored trace)
                continue
            plt.scatter(error_trace_id, y[error_trace_id], marker='*', c='g', s=80)  # mark trace with error
        if highlight_last and i == len(level_list)-1:
            plt.axhline(y=level, color='g', linestyle='-')
        else:
            plt.axhline(y=level, color='r', linestyle='--')
    plt.pause(0.001)
    if highlight_last and save_fig:     #save only last plot
        fig_name = strftime("level_splitting_%Y-%m-%d_%H-%M-%S", gmtime())
        plt.savefig(os.path.join("../out", fig_name))

def preprocess_batch_state(batch_state):
    return batch_state
    batch_state[:, 0] = batch_state[:, 0] / 9
    batch_state[:, 1] = batch_state[:, 1] / 10.5
    batch_state[:, 2] = batch_state[:, 2] / (2*math.pi)
    batch_state[:, 3] = batch_state[:, 3] / 1.5
    batch_state[:, 4] = (batch_state[:, 4] + 0.5) / 10
    batch_state[:, 5] = (batch_state[:, 5] + 0.5) / 11.5
    batch_state[:, 6] = batch_state[:, 6] / (2*math.pi)
    batch_state[:, 7] = batch_state[:, 7] / 1.5
    batch_state[:, 8] = batch_state[:, 8] / 10
    return batch_state

reward_list = collections.deque(maxlen=500)
def plot_reward():
    plt.figure(3)
    plt.title("Recent Episode Reward")
    plt.plot(reward_list, 'b')
    plt.pause(0.001)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

sys = System(err_threshold=0.8)
prefix_list.append(sys.original_s0)     # init prefix list with initial state (i can call only after `sys` init)
model_params = dict()
model_params['batch_size'] = batch_size
model_params['hidden_activation'] = 'leakyrelu'
model = EKFModel(model_params)
replay_memory = SequentialMemory(limit=mem_limit, window_length=mem_window_len)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(lr=0.01)   #large `lr` generate NaN!
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

sys.print_config()
model.print_config()

#IS
max_sim_step=1000000
step_counter = 0
epoch_counter = 0
episode_counter = 0
falsification_flag = False
falsification_counter = 0
while step_counter < max_sim_step:
    # Running step
    while available_indices:
        if step_counter >= max_sim_step:
            break
        episode_counter = episode_counter + 1
        sampled_prefix = rand.sample(prefix_list, 1)[0].copy()  #sample return a list of 1 (k) element (avoid mem problem)
        sys.set_prefix(sampled_prefix)
        sys.run_system()
        trace = sys.get_trace()
        reward = sys.reward
        reward_list.append(reward)
        # learning phase
        for i in range(sys.last_init_state, sys.i_max_reward+1):    #reward until best state
            step_counter = step_counter + 1
            terminal = (i==trace.shape[1]-1)
            observation = trace[:, i]
            #replay_memory_observation.append(observation)
            #replay_memory_reward.append(reward)
            replay_memory.append(observation, None, reward, terminal)  # action is None
            if step_counter >= mem_warmup_steps:    # Learning step
                experience = replay_memory.sample(batch_size)
                batch_state = np.array([exp.state0[0][0:9] for exp in experience])
                #batch_state = np.array([observation[0:9]])
                batch_reward = np.array([exp.reward for exp in experience])
                train_step(batch_state, batch_reward)

        # skip IS warmup episodes
        id = -1     # eventually to detect falsification happened in one of the warmup episodes (lucky shot)
        id = store_current_trace(trace)
        if episode_counter < is_warmup_episodes:
            prefix = compute_random_prefix(id)
            prefix_list.append(prefix)
            available_indices.append(id)
        elif episode_counter == is_warmup_episodes:
            # longer learning step
            #num_learning_steps = 3125
            #for step in range(num_learning_steps):
            #    experience = replay_memory.sample(batch_size)
            #    batch_state = np.array([exp.state0[0][0:9] for exp in experience])
            #    batch_reward = np.array([exp.reward for exp in experience])
            #    train_step(batch_state, batch_reward)
            print("[Info] Start Importance Splitting search")
            reset_importance_splitting()    #finish exploratory phase, start ISplitting
        # falsification detection
        if sys.is_current_trace_false():
            print("[Info] FALSIFICATION!")
            falsification_flag = True
            falsification_counter = falsification_counter + 1
            trace_ids_falsification.append(id)
            sys.render("Episode {}, Steps {} - Reward {}".format(episode_counter, step_counter, reward))
            if step_by_step_exec_flag:
                input()
        episode_steps = trace.shape[1] - sys.last_init_state - 1
        title=template_episode.format(step_counter, max_sim_step, episode_counter, episode_steps, reward, sys.i_max_reward-sys.last_init_state+1)
        print(title)
        #sys.render(title=title)
        #if episode_counter % 10 == 0:
        #    plot_reward()

    if falsification_flag:
        falsification_procedure()
        reset_importance_splitting()
        level = np.NINF  # Current level, since minimization initialize as INF
        level_counter = 0  # current number of levels
        falsification_flag = False  # flag to mark the last iteration of ISplit, when Fals. occurred
        trace_ids_falsification = []
        # Rendering of levels
        level_list = []
        level_prob_list = []  # list of level reach. prob in current exec (each restart, reset)
        continue
    elif step_counter >= max_sim_step:
        # end procedure
        # break all
        print("[Info] End Importance Splitting. Falsification occurred {} times.".format(falsification_counter))
        break

    # learning step
    #shuffle(buffer_size) = mem_limit for perfect shuffle (memory consuming)
    #num_learning_steps = 1250
    #for step in range(num_learning_steps):
    #    experience = replay_memory.sample(batch_size)
    #    batch_state = np.array([exp.state0[0][0:9] for exp in experience])
    #    batch_reward = np.array([exp.reward for exp in experience])
    #    train_step(batch_state, batch_reward)

    # Evaluation step (if no falsification and still steps -> continue IS)
    if episode_counter >= is_warmup_episodes:
        evaluate_all_traces_with_qlearning()
        found_next_level, next_level = choose_next_level()
        if not found_next_level:
            print("[Info] Not found new level.")
            if step_by_step_exec_flag:
                input()
            #reset
            reset_importance_splitting()
            level = np.NINF  # Current level, since minimization initialize as INF
            level_counter = 0  # current number of levels
            falsification_flag = False  # flag to mark the last iteration of ISplit, when Fals. occurred
            trace_ids_falsification = []
            # Rendering of levels
            level_list = []
            level_prob_list = []  # list of level reach. prob in current exec (each restart, reset)
            continue
        # continue evaluation i splitting
        bool_split, num_prom_traces, num_non_prom_traces = split_traces(next_level)  # check each trace if is above or below the current level
        if num_prom_traces == 0:
            # it really happens??? since score>=nextlevel maybe never
            #reset
            import ipdb
            ipdb.set_trace()
            break
        else:  # at least a good trace has been found
            level = next_level
            update_prefix_list_with_new_level_prefixes(bool_split)
            # Store level info
            level_list.append(level)
            level_prob_list.append(num_prom_traces / num_particles)  # Compute prob as avg
            level_counter = level_counter + 1
            print("[Info] {}-TH LEVEL FOUND: {}, Considering {}/{} traces".format(level_counter, level,
                                                                                  num_prom_traces, num_particles))
            # TODO
            plot_score()
            if step_by_step_exec_flag:
                input()
    else:
        # DEPRECATED never here because the index is suddenly added
        # random cut for exploration phase
        #update_prefix_list_with_random_prefixes()
        pass



# import ipdb
# ipdb.set_trace()
