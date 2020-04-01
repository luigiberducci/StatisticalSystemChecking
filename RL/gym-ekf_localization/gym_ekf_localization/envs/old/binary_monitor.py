import system as sys
import numpy as np

def sat_property(trueTrajectory, estimatedTrajectory, timeSteps):
    """
        Property:   Eventually error in the estimate is greater than `epsilon`  (F time>transient & |true-estimate|>epsilon)
        epsilon = 0.6 -> 9/1000 reach error state
    """
    EPSILON = 0.6
    TRANSIENT_TIME = 3.0

    non_zero = np.argwhere((timeSteps>TRANSIENT_TIME) & (np.linalg.norm(trueTrajectory[0:2, :] - estimatedTrajectory[0:2, :], axis=0) > EPSILON))
    result = 0 if len(non_zero)==0 else 1
    return result

if __name__=='__main__':
    pass
