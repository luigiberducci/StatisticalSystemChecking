import argparse as parse
import ipdb
import numpy as np
from statistics  import mean
from matplotlib import pyplot as plt

levels = list()
level_probs = list()
error_probs = list()
axis_steps = list()

def process_version_1(files, groundtruth=0):
    global levels
    global level_probs
    global error_probs
    global axis_steps
    list_num_falsification = list()
    list_num_is_executions = list()
    list_first_fals_steps = list()
    list_first_fals_episodes = list()
    list_est_probabilities = list()
    list_times = list()
    for file in files:
        print("[Info] Processing file {}".format(file.name))
        text = file.readlines()
        levels = list()
        level_probs = list()
        error_probs = list()
        axis_steps = list()
        counter_fals = 0
        first_line_after_falsification = False
        first_falsification_step = None
        first_falsification_episode = None
        first_line_after_complete_is = False
        elapsed_time = None
        mean_n_levels = None
        mean_err_prob = 0
        for line in text:
            if not first_line_after_falsification and counter_fals==1 and not line.startswith("[Info]") and not line.startswith("\n") and not line.startswith("\n", 1):
                #this is the first line after falsification to extract the number of steps/episode in which occurred
                first_line_after_falsification = True
                first_falsification_step, first_falsification_episode = process_first_falsification(line)
            if first_line_after_complete_is and line!='\n': #because the first line after is empty '\n'
                axis_steps.append(process_occurrence_complete_is(line))
                first_line_after_complete_is = False
            if '[Info] FALS' in line:
                counter_fals = counter_fals + 1
            if '[Info] Levels' in line:
                l = process_level_line(line)
                levels.append(process_level_line(line))
            if '[Info] Cond. Prob' in line:
                level_probs.append(process_cond_prob_line(line))
            if '[Info] Error Prob' in line:
                error_probs.append(process_error_prob_line(line))
                first_line_after_complete_is = True
            if 'done, took' in line:
                elapsed_time = process_time_line(line)
        if len(level_probs) > 0:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

            plot_error_probabilities_on_axis(ax1, groundtruth)
            # plot_relative_error_on_error_prob_estimation(ax2, groundtruth)
            # plot_level_distribution_on_axis(ax3)
            plot_number_of_levels_distribution(ax2)

            # textual data
            #for i, (lev, lev_p, err_p) in enumerate(zip(levels, level_probs, error_probs)):
            #    print("Falsification #{}".format(i+1))
            #    print("Levels: " + str(lev))
            #    print("Probs: " + str(lev_p) + " => " + str(err_p))
            #print()
            mean_n_levels = mean([len(l) for l in levels])
            mean_err_prob = mean([err for err in error_probs])
            plt.show()
        print("Summary - Num Fals: {}, Num ISplit Iters: {}, First Fals: {} step, {} episode, Mean num Levels: {}, Mean Err Prob: {}, Time: {}".format(
                                                                                                            counter_fals,
                                                                                                            len(error_probs),
                                                                                                            first_falsification_step,
                                                                                                            first_falsification_episode,
                                                                                                            mean_n_levels,
                                                                                                            mean_err_prob,
                                                                                                            elapsed_time))
        list_num_falsification.append(counter_fals)
        list_num_is_executions.append(len(error_probs))
        if first_falsification_step is not None:
            list_first_fals_steps.append(first_falsification_step)
            list_first_fals_episodes.append(first_falsification_episode)
        list_est_probabilities.append(mean_err_prob)
        list_times.append(elapsed_time)
    print("")
    print("AVG - Num Fals: {}, Num ISplit Iters: {}, First Fals: {} step, {} episode, Mean num Levels: {}, Mean Err Prob: {}, Time: {}".format(
                                                                                                            mean(list_num_falsification),
                                                                                                            mean(list_num_is_executions),
                                                                                                            mean(list_first_fals_steps),
                                                                                                            mean(list_first_fals_episodes),
                                                                                                            None,
                                                                                                            mean(list_est_probabilities),
                                                                                                            mean(list_times)))

def process_model_weights(directory):
    import os
    model_dir = os.path.join(directory, "models")
    for weight_file in os.listdir(model_dir):
        model = create_the_same_model()
        if model is None:
            return

def create_the_same_model():
    #TODO
    return None

def plot_relative_error_on_error_prob_estimation(ax, groundtruth=0.5**10):
    rel_errors = [abs((groundtruth-err)/groundtruth) for err in error_probs]
    ax.plot(rel_errors)
    ax.set_xlabel("Learning process")
    ax.set_ylabel("Relative error")
    # ipdb.set_trace()

def plot_number_of_levels_distribution(ax2):
    # y = [len(l) if len(l)>1 else -1 for l in levels]
    yy = [len(l) for l in levels]
    ax2.scatter(axis_steps, yy)
    ax2.set_title("Adaptive Levels")
    ax2.set_xlabel("Learning process")
    ax2.set_ylabel("Number of levels")

def plot_error_probabilities_on_axis(ax1, groundtruth=0.5**10):
    error_probs_simplified = list()
    for i, (err, step) in enumerate(zip(error_probs, axis_steps)):
        if len(levels[i]) > 5:
            ax1.scatter(step, err, marker='o')
        else:
            ax1.scatter(step, err, marker='X')
        error_probs_simplified.append(err)
    ax1.plot(axis_steps, error_probs)
    ax1.axhline(y=groundtruth, color='r', linestyle='--')
    ax1.set_title("Error Probability Estimation")
    ax1.set_xlabel("Learning process")
    ax1.set_ylabel("Estimated Error Probability")

def plot_level_distribution_on_axis(ax2):
    max_number_levels = compute_max_num_levels(levels)
    xs = np.arange(len(levels))
    for i in range(max_number_levels):
        i_levels = np.array([lst[i] if len(lst) > i else None for lst in levels]).astype(np.double)
        s_mask = np.isfinite(i_levels)
        ax2.plot(xs[s_mask], i_levels[s_mask])
    ax2.legend(np.arange(1, max_number_levels + 1))  # count level starting by 1
    ax2.set_title("Levels Distribution")

def compute_max_num_levels():
    return max([len(l) for l in levels])

def process_occurrence_complete_is(line):
    #return the occurrence (#step) when the is split has been completed
    line = line.split(',')[0]
    line = line.split(': episode: ')
    return int(line[0].split('/')[0]) - 1   #-1 because this is the first line AFTER the complete is

def process_time_line(line):
    return float(line.split(' ')[2])

def process_first_falsification(line):
    #line format: `    step/tot_step: episode: num_ep, duration: ...`
    line = line.split(',')[0]
    line = line.split(': episode: ')
    return int(line[0].split('/')[0])-1, int(line[1])-1     #-1 because this is the first line AFTER falsification

def process_level_line(line):
    line = line.replace("[Info] ", "")  #remove header
    init = line.find('[')   #find level list
    ende = line.find(']')
    assert init >= 0 & ende >= 0    #it must exist
    sub = line[init+1: ende]    #in python last index is not inclusive
    return [float(val) for val in sub.split(', ')]

def process_cond_prob_line(line):
    line = line.replace("[Info] ", "")  # remove header
    init = line.find('[')  # find level list
    ende = line.find(']')
    assert init >= 0 & ende >= 0  # it must exist
    sub = line[init + 1: ende]  # in python last index is not inclusive
    return [float(val) for val in sub.split(', ')]

def process_error_prob_line(line):
    line = line.replace("[Info] ", "")  # remove header
    init = line.find(': ')  # find level list
    assert init >= 0
    sub = line[init+1:-1]  # -1 to remove newline
    return float(sub)

def main():
    parser = parse.ArgumentParser("Process log files")
    parser.add_argument('files', type=parse.FileType('r'), nargs='+',
                        help='log files to process')
    parser.add_argument('--groundtruth', type=float, nargs='?', help='Groundtruth of error probability', default=0.5**10)
    args = parser.parse_args()
    files = args.files
    gt = args.groundtruth
    process_version_1(files, groundtruth=gt)

if __name__=="__main__":
    main()
