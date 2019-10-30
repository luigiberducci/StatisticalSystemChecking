import system as sys
import numpy as np
import ipdb
import matplotlib.pyplot as plt

def monitor(trueTrajectory, estimatedTrajectory, timeSteps):
    """
        Property:   Eventually error in the estimate is greater than `epsilon`  (F time>transient & |true-estimate|>epsilon)
        epsilon = 0.6 -> 9/1000 reach error state
    """
    EPSILON = 0.6
    TRANSIENT_TIME = 3.0

    # We can model the property as F (-transient & error)
    trans = np.where(timeSteps > TRANSIENT_TIME, 1, 0)
    error = np.where(np.linalg.norm(trueTrajectory[0:2, :] - estimatedTrajectory[0:2, :], axis=0) > EPSILON, 1, 0)
    non_zero = np.argwhere((trans==1) & (error==1))
    first_one = len(trans)+1 if len(non_zero)==0 else non_zero[0][0]
    monitor0 = np.zeros(first_one-1)    # sequence of state 0
    monitor1 = np.ones(len(trans)-first_one+1)  # sequence of state 1
    monitor = np.concatenate([monitor0, monitor1])  #sequence of discrete state of DFA
    # print(trueTrajectory[0:2, :].shape)
    # print(estimatedTrajectory[0:2, :])
    return np.vstack([trueTrajectory, estimatedTrajectory, monitor])

def main():
    for i in range(1000):
        hxTrue, hxTime, hxEKF = sys.run_system()
        full = monitor(hxTrue, hxEKF, hxTime)
        if full[8][101]==1:
            plt.plot(np.transpose(full[0:2, :]), "-g")
            plt.plot(np.transpose(full[4:6, :]), "-r")
            plt.plot(np.transpose(full[8, :]), "-b")
            # ipdb.set_trace()
            # plt.show()
        # print(full)

if __name__=="__main__":
    main()
