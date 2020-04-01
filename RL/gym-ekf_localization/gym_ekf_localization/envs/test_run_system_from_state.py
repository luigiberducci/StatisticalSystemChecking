import ipdb
from EKFLocEnv import System
import numpy as np
import matplotlib.pyplot as plt

def test_run_system_from_state():
    sys = System()
    true, est, time = sys.run_system()
    i = 25
    trueState = np.reshape(true[:, i], (4, 1))
    estState = np.reshape(est[:, i], (4, 1))
    t = time[i]
    true2, est2, time2 = sys.run_system_from_state(trueState, estState, t)

    i = 50
    trueState = np.reshape(true[:, i], (4, 1))
    estState = np.reshape(est[:, i], (4, 1))
    t = time[i]
    true3, est3, time3 = sys.run_system_from_state(trueState, estState, t)

    i = 75
    trueState = np.reshape(true[:, i], (4, 1))
    estState = np.reshape(est[:, i], (4, 1))
    t = time[i]
    true4, est4, time4 = sys.run_system_from_state(trueState, estState, t)

    #plt.plot(np.transpose(true[0:2, :]), np.transpose(est[0:2, :]))
    plt.cla()

    plt.plot(true[0, :].flatten(),
             true[1, :].flatten(), "-r")

    plt.plot(est[0, :].flatten(),
             est[1, :].flatten(), "-b")
    plt.plot(est2[0, :].flatten(),
             est2[1, :].flatten(), "-g")
    plt.plot(est3[0, :].flatten(),
             est3[1, :].flatten(), "-y")
    plt.plot(est4[0, :].flatten(),
             est4[1, :].flatten(), "-o")

    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.001)
    plt.show()

if __name__=='__main__':
    test_run_system_from_state()
