import argparse
import os
import sys

from itertools import product
from time import strftime, gmtime

from RLISAgent import RLISAgent
from model.TRModel import TRModel
from model.EKFModel import EKFModel
from model.SRModel import SRModel
from model.CRModel import CRModel
from rl.memory import SequentialMemory
from env.EKF import EKFSystem
from env.SR import SRSystem
from env.TR import TRSystem
from env.CROSS import CrossRoadEnv
from timeit import default_timer as timer

# Templates
template_config = "Configuration:\n" \
                  "Problem: {}, Params: {}\n" \
                  "Memory: limit: {}, window: {}, mem warmup steps: {}\n" \
                  "Model: batch: {}, hidden init: {}, hidden act: {}, out act: {}\n" \
                  "Agent: optimizer: {}, lr: {}, params: {}, loss: {}\n" \
                  "ISplit: max sim steps: {}, particles: {}, k particles: {}\n"

# ex: out/EKF/1000000/cust_pref_datetime_hidinit_gl_hidact_relu_batch_8_mem_20000_500_opt_sgd_lr_0_01_isplit_n_200_k_10/1
out_dir_template = "out_prod/{}/{}/{}{}_rscale_{}_statevars_{}_hidinit_{}_hidact_{}_batch_{}_mem_{}_{}_opt_{}_lr_{}_isplit_n_{}_k_{}_d_{}/{}"

template_training_phase = "[Info] Training phase completed in {} seconds. \n" \
                          "[Result] Num Fals: {}, Num IS iters: {}, First Fals: {}, Mean Error Probs: {}\n" \
                          "[Details] Error Probs: {}"
template_test_phase = "[Info] Testing phase completed in {} seconds. Num Fals: {}, Error Probs: {}"

def get_default_sys(problem_name):
    if problem_name == 'EKF':
        err_threshold = 0.8
        sys = EKFSystem(err_threshold=err_threshold)
    elif problem_name == 'SR':
        p = 0.2
        sys = SRSystem(p=p)
    elif problem_name == 'TR':
        sys = TRSystem()
    elif problem_name == 'CR':
        sys = CrossRoadEnv()
    else:
        raise ValueError("problem name {} is not defined".format(problem_name))
    return sys

def get_default_memory_configuration(problem_name):
    mem_window = 1
    if problem_name == 'EKF':
        mem_limit = 200000
        mem_warmup_steps = 1000
    elif problem_name == 'SR':
        mem_limit = 10000
        mem_warmup_steps = 500
    elif problem_name == 'TR':
        mem_limit = 200000
        mem_warmup_steps = 1000
    elif problem_name == 'CR':
        mem_limit = 100000
        mem_warmup_steps = 1000
    else:
        raise ValueError("problem name {} is not defined".format(problem_name))
    return mem_limit, mem_window, mem_warmup_steps

def get_default_model_configuration():
    batch_size = 8
    out_activation = 'linear'
    hidden_init = 'glorot_uniform'
    hidden_activation = 'leakyrelu'
    return batch_size, hidden_init, hidden_activation, out_activation

def get_best_model_files(problem_name, rob_scaling=True):
    model_files = []
    scaling = [1] * 5
    if problem_name == 'EKF':
        if rob_scaling:
            directory = "out_prod/EKF/1000000/2020-04-02_19-56-15_rscale_True_statevars_7_hidinit_glorot_uniform_hidact_leakyrelu_batch_32_mem_200000_1000_opt_sgd_lr_0.01_isplit_n_100_k_10_d_0.0"
            files = ["1/models/weights_1000006.h5", "2/models/weights_1000098.h5", "3/models/weights_1000018.h5", "4/models/weights_1000038.h5", "5/models/weights_1000008.h5"]
            scaling = [0.3957554479374316, 0.3931015621627058, 0.3958666413777874, 0.3975290434643983, 0.3946757717345769]
        else:
            directory = "out_prod/EKF/1000000/2020-04-08_18-43-29_rscale_False_statevars_7_hidinit_glorot_uniform_hidact_leakyrelu_batch_32_mem_200000_1000_opt_sgd_lr_0.01_isplit_n_100_k_10_d_0.01"
            files = ["1/models/weights_1000091.h5", "2/models/weights_1000056.h5", "3/models/weights_1000032.h5", "4/models/weights_1000034.h5", "5/models/weights_1000009.h5"]
    elif problem_name == 'SR':
        if rob_scaling:
            directory = "out_prod/SR/100000/2020-04-01_16-28-46_rscale_True_statevars_2_hidinit_glorot_uniform_hidact_leakyrelu_batch_64_mem_10000_500_opt_sgd_lr_0.01_isplit_n_100_k_10_d_0.0/"
            files = ["1/models/weights_100000.h5", "2/models/weights_100000.h5", "3/models/weights_100003.h5", "4/models/weights_100002.h5", "5/models/weights_100005.h5"]
            scaling = [8.009845986984738, 7.97947722342725, 8.020691973969551, 8.029368763557404, 8.007676789587773]
        else:
            directory = "out_prod/SR/100000/2020-04-08_14-36-38_rscale_False_statevars_2_hidinit_glorot_uniform_hidact_leakyrelu_batch_64_mem_10000_500_opt_sgd_lr_0.01_isplit_n_100_k_10_d_0.01"
            files = ["1/models/weights_100002.h5", "2/models/weights_100000.h5", "3/models/weights_100006.h5", "4/models/weights_100004.h5", "5/models/weights_100005.h5"]
    elif problem_name == 'TR':
        if rob_scaling:
            directory = "out_prod/TR/1000000/2020-04-05_03-56-44_rscale_True_statevars_8_hidinit_glorot_uniform_hidact_leakyrelu_batch_32_mem_200000_1000_opt_sgd_lr_0.01_isplit_n_100_k_10_d_0.01"
            files = ["1/models/weights_1000044.h5", "2/models/weights_1000009.h5", "3/models/weights_1000038.h5", "4/models/weights_1000046.h5", "5/models/weights_1000033.h5"]
            scaling = [0.15789567868309373, 0.15686755578734904, 0.15668854916784367, 0.15721972332028536, 0.15912140467027291]
        else:
            directory = "out_prod/TR/1000000/2020-04-08_22-10-53_rscale_False_statevars_8_hidinit_glorot_uniform_hidact_leakyrelu_batch_32_mem_200000_1000_opt_sgd_lr_0.01_isplit_n_100_k_10_d_0.01"
            files = ["1/models/weights_1000056.h5", "2/models/weights_1000029.h5", "3/models/weights_1000015.h5", "4/models/weights_1000009.h5", "5/models/weights_1000007.h5"]
    else:
        raise ValueError("problem name {} is not defined".format(problem_name))
    return [os.path.join(directory, mfile) for mfile in files], scaling

def get_default_model(problem_name, batch_size, hidden_init, hidden_activation, out_activation, ninputs):
    model_manager = None
    if problem_name == 'EKF':
        model_manager = EKFModel(batch_size, hidden_init, hidden_activation, out_activation, ninputs)
    elif problem_name == 'SR':
        model_manager = SRModel(batch_size, hidden_init, hidden_activation, out_activation, ninputs)
    elif problem_name == 'TR':
        model_manager = TRModel(batch_size, hidden_init, hidden_activation, out_activation, ninputs)
    elif problem_name == 'CR':
        model_manager = CRModel(batch_size, hidden_init, hidden_activation, out_activation, ninputs)
    else:
        raise ValueError("problem name {} is not defined".format(problem_name))
    return model_manager, model_manager.get_model()


def get_trained_model(problem_name, model_file, batch_size, hidden_init, hidden_activation, out_activation):
    import tensorflow as tf
    model_manager = None
    if problem_name == 'EKF':
        ninputs = 7
        model_manager = EKFModel(batch_size, hidden_init, hidden_activation, out_activation, ninputs)
    elif problem_name == 'SR':
        ninputs = 2
        model_manager = SRModel(batch_size, hidden_init, hidden_activation, out_activation, ninputs)
    elif problem_name == 'TR':
        ninputs = 8
        model_manager = TRModel(batch_size, hidden_init, hidden_activation, out_activation, ninputs)
    elif problem_name == 'CR':
        model_manager = CRModel(batch_size, hidden_init, hidden_activation, out_activation, ninputs)
    else:
        raise ValueError("problem name {} is not defined".format(problem_name))
    return model_manager, tf.keras.models.load_model(model_file)

def get_default_training_params(problem_name):
    if problem_name == 'EKF':
        max_sim_steps = 10000 * 100  # EKF
        num_particles = 100
        k_particles = 10
        delta = 0.0
    elif problem_name == 'SR':
        max_sim_steps = 1000 * 10  # SR
        num_particles = 200
        k_particles = 10
        delta = 0.00
    elif problem_name == 'TR':
        max_sim_steps = 10000 * 65 # TR
        num_particles = 200
        k_particles = 10
        delta = 0.01
    elif problem_name == 'CR':
        max_sim_steps = 10000 * 152 # CR
        num_particles = 500
        k_particles = 10
        delta = 0.01
    else:
        raise ValueError("problem name {} is not defined".format(problem_name))
    return max_sim_steps, num_particles, k_particles, delta

def training_phase(sys, agent, max_sim_steps, num_particles, k_particles, delta, render, rscale_flag):
    print("[Info] Training...")
    # Exploration with linear particles decay
    min_n = num_particles
    max_n = num_particles
    inc = 0
    expl_steps = 5000
    # Train
    start_time = timer()
    error_prob_list, num_fals, first_fals_step = agent.train(sys, max_sim_steps=max_sim_steps, render=render,
                                                             max_num_particles=max_n, min_num_particles=min_n, particles_inc = inc,
                                                             k_particles=k_particles, exploratory_steps=expl_steps,
                                                             delta=delta, rscale_flag=rscale_flag)
    elapsed_time = timer() - start_time
    # Log info
    mean_error_probs = 0 if len(error_prob_list) == 0 else sum(error_prob_list) / len(error_prob_list)
    print(template_training_phase.format(elapsed_time, num_fals, len(error_prob_list), first_fals_step,
                                         mean_error_probs, error_prob_list))

def testing_phase(sys, agent, num_particles, k_particles, delta, render, num_test=10, rscale_flag=True, scale_factor=1):
    error_prob_list, sum_num_fals = [], 0
    print("[Info] Offline Testing with N={}, K={}, Delta={}. Num iterations={}".format(num_particles, k_particles, delta, num_test))
    gstart = timer()
    for test in range(num_test):
        start = timer()
        error_prob, num_falsifications, _ = agent.test(sys, render=render, num_particles=num_particles,
                                                       k_particles=k_particles, delta=delta, rscale_flag=rscale_flag, scale_factor=scale_factor)
        error_prob_list.append(error_prob)
        sum_num_fals = sum_num_fals + num_falsifications
        elapsed_time = timer() - start
        print(template_test_phase.format(elapsed_time, num_falsifications, error_prob))
    gelapsed = timer() - gstart
    mean_error_probs = 0 if len(error_prob_list) == 0 else sum(error_prob_list) / len(error_prob_list)
    print("[Info] Testing phase completed in {} seconds.".format(gelapsed))
    print("[Result] Num Fals: {}, Num IS iters: {}, Mean Error Probs: {}".format(sum_num_fals, num_test, mean_error_probs))
    print("[Details] Error Probs: {}".format(error_prob_list))

def run(problem_name, mem_limit, mem_warmup_steps, batch_size, hidden_init, hidden_activation,
        out_activation, optimizer, lr, opt_params, loss, max_sim_steps,
        num_particles, k_particles, delta, enable_test_flag, out_dir, render, n_inputs=None, rscale_flag=True):
    mem_window = 1  # fixed window for exp replay
    # Create out dir and subdirectories
    level_dir = os.path.join(out_dir, "levels")
    trace_dir = os.path.join(out_dir, "traces")
    model_dir = os.path.join(out_dir, "models")
    for dir in [level_dir, trace_dir, model_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Write config file
    with open(os.path.join(out_dir, "config.txt"), "w+") as f:
        problem_params = "default"
        config_text = template_config.format(problem_name, problem_params, mem_limit, mem_window, mem_warmup_steps,
                                        batch_size, hidden_init, hidden_activation, out_activation,
                                        optimizer, lr, opt_params, loss,
                                        max_sim_steps, num_particles, k_particles)
        f.write(config_text)
    # Initialization sys and memory
    sys = get_default_sys(problem_name)
    memory = SequentialMemory(limit=mem_limit, window_length=mem_window)

    if enable_test_flag:    # Testing
        rscale_flag = True # MANUALLY CHANGE THIS FLAG
        modelfiles, scaling_factors = get_best_model_files(problem_name, rscale_flag)
        for mfile, scale_factor in zip(modelfiles, scaling_factors):
            model_manager, model = get_trained_model(problem_name, mfile, batch_size, hidden_init, hidden_activation, out_activation)
            agent = RLISAgent(model_manager, model, memory, mem_warmup_steps, opt=optimizer, lr=lr, opt_params=opt_params, loss_name=loss, level_dir=level_dir, trace_dir=trace_dir, model_dir=model_dir)
            # Log info
            print("[Info] Start offline testing, model: {}".format(mfile))
            testing_phase(sys, agent, num_particles, k_particles, delta=delta, render=render, num_test=50, rscale_flag=rscale_flag, scale_factor=scale_factor)
    else:                   # Training
        model_manager, model = get_default_model(problem_name, batch_size, hidden_init, hidden_activation, out_activation, n_inputs)
        agent = RLISAgent(model_manager, model, memory, mem_warmup_steps, opt=optimizer, lr=lr, opt_params=opt_params, loss_name=loss, level_dir=level_dir, trace_dir=trace_dir, model_dir=model_dir)
        # Log info
        sys.print_config()
        model_manager.print_config()
        agent.print_config()
        training_phase(sys, agent, max_sim_steps, num_particles, k_particles, delta=0.0, render=render, rscale_flag=rscale_flag)

def run_default(problem_name, out_dir, render, offline_test=False):
    enable_test_flag = offline_test
    # Default init
    mem_limit, mem_window, mem_warmup_steps = get_default_memory_configuration(problem_name)
    max_sim_steps, num_particles, k_particles, delta = get_default_training_params(problem_name)
    batch_size, hidden_init, hidden_activation, out_activation = 8, "glorot_uniform", "relu", "linear"
    optimizer, lr, opt_params, loss = "sgd", 0.0001, [], "mse"
    # Run
    run(problem_name, mem_limit, mem_warmup_steps, batch_size, hidden_init, hidden_activation, out_activation,
        optimizer, lr, opt_params, loss, max_sim_steps, num_particles, k_particles, delta, enable_test_flag, out_dir, render)

def main():
    problems = ['EKF', 'SR', 'TR', 'CR']
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default=["SR"], nargs=1, help='Problem name', choices=problems)
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-offline', action='store_true')
    parser.add_argument('-render', action='store_true')
    parser.add_argument('--outdir', default='out', nargs='?', help='Output directory for results')
    parser.add_argument('--outpref', default='', nargs='?', help='Output directory prefix for testing phase')
    args = parser.parse_args()
    problem_name = args.problem[0]
    if not args.test:
        out_dir = args.outdir
        run_default(problem_name, out_dir, args.render, args.offline)
    else:                       # multi testing
        out_pref = args.outpref
        multi_test(problem_name, out_prefix=out_pref, render=args.render)

def offline_test(problem_name, render=False):
    enable_test_flag = True
    # Default init
    mem_limit, mem_window, mem_warmup_steps = get_default_memory_configuration(problem_name)
    max_sim_steps, num_particles, k_particles, delta = get_default_training_params(problem_name)
    batch_size, hidden_init, hidden_activation, out_activation = 8, "glorot_uniform", "relu", "linear"
    optimizer, lr, opt_params, loss = "sgd", 0.001, [], "mse"
    # Run
    run(problem_name, mem_limit, mem_warmup_steps, batch_size, hidden_init, hidden_activation, out_activation,
        optimizer, lr, opt_params, loss, max_sim_steps, num_particles, k_particles, delta, enable_test_flag, out_dir, render)

def multi_test(problem_name, out_prefix="", render=False):
    num_repeat = 5
    opts = ["sgd"]
    losses = ["mse"]
    lrs = [0.01]
    max_steps = [1000000]
    ns = [200]
    ks = [10]
    deltas = [0.01]
    inits = ["glorot_uniform"]
    acts = ["leakyrelu"]
    out_acts = ["linear"]
    mem_limits = [200000]
    mem_wups = [1000]
    # mem_limits = [10000]    #SR
    # mem_wups = [500]        #SR
    num_input_vars = [8]
    rscale_flags = [False]
    batch_szs = [32]

    # default params
    enable_test_flag = False
    # multi test for each combination of parameter lists
    for combination in product(opts, lrs, losses, max_steps, ns, ks, deltas, num_input_vars, inits, acts, out_acts, mem_limits, mem_wups, batch_szs, rscale_flags):
        opt, lr, loss, max_sim_steps, num_parts, k_parts, delta, ninputs, hid_init, hid_act, out_act, mem_lim, mem_wup, batch_sz, rscale = combination

        date = strftime("%Y-%m-%d_%H-%M-%S", gmtime())  # fixed for all repeatitions
        for repeat in range(num_repeat):
            # create output directory
            out_dir = out_dir_template.format(problem_name, max_sim_steps, out_prefix, date, rscale, ninputs, hid_init, hid_act, batch_sz, mem_lim, mem_wup, opt, lr, num_parts, k_parts, delta, repeat+1)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # store stdout and stderr in log files
            out_log = os.path.join(out_dir, "log.txt")
            err_log = os.path.join(out_dir, "err.txt")
            sys.stdout = open(out_log, 'w')
            sys.stderr = open(err_log, 'w')
            # run RLIS
            run(problem_name, mem_lim, mem_wup, batch_sz, hid_init, hid_act, out_act,
                opt, lr, [], loss, max_sim_steps, num_parts, k_parts, delta, enable_test_flag, out_dir, render,
                ninputs, rscale)

if __name__=="__main__":
    main()
