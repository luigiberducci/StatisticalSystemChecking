import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random as rnd
import ipdb

# ipdb.set_trace()

class SystemSimulator:
    def __init__(self, monitor):
        self.states = np.array(["x0", "x1", "x2"])
        self.i_init_state = 0
        # MC from paper
        self.time_horizon = 5
        self.P = np.array([[0, 1, 0], [0.7, 0, 0.3], [0.5, 0, 0.5]])
        self.monitor = monitor
        # Success Runs
        # self.time_horizon = 2
        # self.P = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0, 0, 1]])    #dummy flip coin

    def get_num_states(self):
        return len(self.states)

    def step(self, x):
        distr = np.cumsum(self.P[x])
        choice = rnd.random()
        return np.argwhere(distr>choice)[0][0] #Extract the value from 1-dim np array

    def simulate_steps(self, x0, steps):
        x = x0
        trace = [x0]
        for i in range(steps):
            x = self.step(x)
            trace.append(x)
        return trace

    def simulate_default(self):
        x0 = self.i_init_state
        return self.simulate_steps(x0, self.time_horizon)

    def create_compound_trace(self, trace):
        # Here run parallely a DFA for specification
        dfa_trace = self.monitor.run_simulation(trace)
        return list(zip(trace, dfa_trace))

    def monitor_success_runs(self, trace):
        # DEPRECATED
        #Implement extension with absorbing states 'succ', 'fail'
        if trace[2] == 2:
            return trace[:3] + [self.succ_state] * (len(trace)-2)
        for i in range(2):
            if trace[i] + 1 != trace[i+1]:
                return trace[:i+1] + [self.fail_state] * (len(trace)-i)

    def run_simulation(self):
        return self.create_compound_trace(self.simulate_default())

class Monitor:
    """
        It describe the specification with a DFA. Spec: F<=3 x==3
    """
    def __init__(self):
        self.states = np.array(["s0", "s1", "s2", "s3", "succ", "fail"])
        self.i_init_state = 0
        self.i_succ_state = 4
        self.i_fail_state = 5

    def step(self, i_s, x):
        """ Next state with index `is` deterministic according to event `x`"""
        if i_s == 0:
            n_x = 1
            if x == 2:
                n_x = self.i_succ_state
        elif i_s == 1:
            n_x = 2
            if x == 2:
                n_x = self.i_succ_state
        elif i_s == 2:
            n_x = 3
            if x == 2:
                n_x = self.i_succ_state
        elif i_s == 3:
            n_x = self.i_fail_state
            if x == 2:
                n_x = self.i_succ_state
        else:
            n_x = i_s   #Else they are `succ` or `fail`, absorbing states
        return n_x

    def run_simulation(self, trace):
        """ Simulation of DFA multi-step. """
        i_s = self.i_init_state
        dfa_trace = [i_s]
        for x in trace:
            i_s = self.step(i_s, x)
            dfa_trace.append(i_s)
        return dfa_trace

class QL:
    def __init__(self, system, monitor, epsilon, max_num_iters, delay_m):
        #Assume complete knowledge of state space, augmented with 'succ' and 'fail'
        self.sys_num_states = len(system.states)    # states from 0 to num_states-1 are the system states
        self.mon_num_states = len(monitor.states)   # states from 0 to num_states-1 are the system states
        self.init_state = (system.i_init_state, monitor.i_init_state)            # init state is in pos 0
        self.i_succ_state = monitor.i_succ_state  #succ states are all the state with (_,s) s == monitor.i_succ_state
        self.i_fail_state = monitor.i_fail_state  #as the line above, but with i_fail_state
        # Create U (upperbound) and L (lowerbound) tables
        self.U = np.ones((self.mon_num_states, self.sys_num_states))
        self.L = np.zeros((self.mon_num_states, self.sys_num_states))
        self.U[self.i_fail_state] = 0   # prob. that fail state will reach succ is 0, in all exit transitions
        self.L[self.i_succ_state] = 1   # prob. that succ state will reach succ is 1, in all exit transitions
        # Create accumulator (ACC) for U,L
        self.accumU = np.zeros((self.mon_num_states, self.sys_num_states))    #all the states have accumulators
        self.accumL = np.zeros((self.mon_num_states, self.sys_num_states))
        self.counter = np.zeros((self.mon_num_states, self.sys_num_states))  #all the states have counter
        self.learn_flag = np.full((self.mon_num_states, self.sys_num_states), True)    #init learn flag true for all states
        self.last_updates = np.zeros((self.mon_num_states, self.sys_num_states))
        self.last_update = 0
        self.update_counter = 0
        # Learner parameters
        self.epsilon = epsilon          # approx precision
        self.max_iters = max_num_iters  # max number of iterations
        self.m = delay_m                # number of visits before apply q-value update

    def learning(self, sys):
        """
            Interval Iteration Algorithm
            input:  MDP with no end-components (EC)
            output: probability of reach 'succ'
        """
        remaining_iters = self.max_iters
        while remaining_iters>0:
            if remaining_iters % 10000 == 0:
                print("[Info] Remaining iters: {} | U-L Difference S0: {}".format(remaining_iters, self.U[self.init_state] - self.L[self.init_state]))
            #Explore phase
            trace = sys.run_simulation()

            #Update phase
            # ipdb.set_trace()
            t = len(trace) - 1
            while t>0:
                s1 = trace[t]
                s  = trace[t-1]
                # Update U, L
                self.update(s, s1, t)
                t = t-1;  #Pop last element
            # print(trace)
            # print(self.U)
            # print(self.L)
            # print()
            # Check for convergence
            error = self.U[self.init_state] - self.L[self.init_state]
            if error < 2*self.epsilon:
                # ipdb.set_trace()
                print("[Info] Probabilistic reachability converged")
                print("[Result] U-L Diff in S0: {} | Approx P(F succ): {}".format(error, (self.U[self.init_state]+self.L[self.init_state])/2))
                break
            remaining_iters = remaining_iters-1
        if remaining_iters==0:
            print("[Info] Reached max num iterations ({})".format(self.max_iters))

    def update(self, state, state1, t):
        x, s = state        #unpack state in system state and monitor state
        x1, s1 = state1
        # Check if the state is a terminal one
        if s in [self.i_succ_state, self.i_fail_state]:
            return              #eventually no update

        # if self.learn_flag[s] == True:
        if True:
            self.accumU[s][x] = self.accumU[s][x] + self.U[s1][x1]
            self.accumL[s][x] = self.accumL[s][x] + self.L[s1][x1]
            self.counter[s][x] = self.counter[s][x] + 1
            if self.counter[s][x] == self.m:
                #Update U-value
                accum_avg = self.accumU[s][x] / self.m
                if accum_avg <= (self.U[s][x] - 2*self.epsilon):
                    self.U[s][x] = accum_avg + self.epsilon
                    self.last_update = t
                elif self.last_updates[s][x] >= self.last_update:
                    self.learn_flag[s][x] = False
                #Update L-value
                accum_avg = self.accumL[s][x] / self.m
                if accum_avg >= (self.L[s][x] + 2*self.epsilon):
                    self.L[s][x] = accum_avg - self.epsilon
                    self.last_update = t
                elif self.last_updates[s][x] >= self.last_update:
                    self.learn_flag[s][x] = False
                self.accumU[s][x] = 0
                self.accumL[s][x] = 0
                self.counter[s][x] = 0 #reset counter of state s
                self.last_updates[s][x] = t
        elif self.last_updates[s][x] < self.last_update:
            self.learn_flag[s][x] = True

def main():
    mon = Monitor()
    sys = SystemSimulator(mon)
    # for i in range(10):
    #     print(sys.run_simulation())

    # Define QLearner. Set index of init state as `0`, then agent parameters
    agent = QL(sys, mon, epsilon=0.01, max_num_iters=100000, delay_m=100)
    e_time = time.time()
    agent.learning(sys)
    e_time = time.time() - e_time
    print("U-Table")
    print(agent.U)
    print("L-Table")
    print(agent.L)
    print("\n[Info] Elapsed time {}".format(e_time))
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(np.arange(0, len(sys.states), 1), np.arange(0, len(mon.states), 1))
    u = ax.plot_surface(X, Y, agent.U, cmap=cm.coolwarm, vmin=0, vmax=1)
    l = ax.plot_surface(X, Y, agent.L, cmap=cm.coolwarm, vmin=0, vmax=1)
    ax.set_title('U-value and L-value')
    fig.colorbar(u, shrink=0.5, aspect=5)
    plt.show()

if __name__=="__main__":
    for i in range(10):
        main()
