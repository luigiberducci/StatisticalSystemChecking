from env.EKF import EKFSystem
from env.TR import TRSystem
import argparse
import multiprocessing as mp
import random
import matplotlib.pyplot as plt

def get_system(problem):
    if problem == "EKF":
        return EKFSystem()
    elif problem == "TR":
        return TRSystem()
    else:
        raise ValueError("problem {} not implemented".format(problem))

def core_func(p, sys, sims):
    template_log = "[Process {}] episode: {}/{}, collisions: {}, reward: {}"
    random.seed(p)
    for i in range(sims):
        sys.reset_init_state()
        sys.run_system()
        collision = sys.reward > 1
        plt.figure(p+1)
        sys.render()
        print(template_log.format(p, i+1, sims, collision, sys.reward))

parser = argparse.ArgumentParser()
parser.add_argument("--problem", default="EKF", help="problem name", choices=["EKF", "TR"])
parser.add_argument("--cores", default=2, help="number of dedicated cores", nargs='?')
parser.add_argument("-n", default=2, help="number of simulations")
args = parser.parse_args()
problem, n, cores = args.problem, args.n, args.cores

nn = n // cores             # num sims for each core
nn_last = nn + n % cores     # num sims for the last core

processes = []
for i in range(cores):
    sys = get_system(problem)
    sims = nn
    if i == cores-1:
        sims = nn_last
    process = mp.Process(target=core_func, args=(i, sys, sims))
    processes.append(process)

# Fork
for p in processes:
    p.start()
# Join
for p in processes:
    p.join()

import ipdb
ipdb.set_trace()
