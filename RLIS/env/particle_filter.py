"""

Particle Filter localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import math

import matplotlib.pyplot as plt
import numpy as np

# Estimation parameter of PF
Q = np.diag([0.2]) ** 2  # range error

#  Simulation parameter
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 10.0  # simulation time [s]

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

show_animation = True
time_event = 5.0

def calc_input(time):
    v = 1.5  # [m/s]
    yawrate = 0.35  # [rad/s]
    if time >= time_event:
        yawrate = -0.35  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    ## add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    return xTrue, z, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F.dot(x) + B.dot(u)

    return x

def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    z = H @ x
    return z

def gauss_likelihood(x, sigma):
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p


def calc_covariance(xEst, px, pw):
    cov = np.zeros((3, 3))

    for i in range(px.shape[1]):
        dx = (px[:, i] - xEst)[0:3]
        cov += pw[0, i] * dx.dot(dx.T)
    cov /= NP

    return cov


def pf_localization(px, pw, z, u):
    """
    Localization with Particle filter
    """

    for ip in range(NP):
        x = np.array([px[:, ip]]).T
        w = pw[0, ip]

        #  Predict with random input sampling
        ud = u + 1.5*INPUT_NOISE @ np.random.randn(2, 1)
        x = motion_model(x, ud)

        #  Calc Importance Weight

        for i in range(len(z[:, 0])):
            dx = x[0, 0] - z[0, 0]
            dy = x[1, 0] - z[1, 0]
            dz = math.sqrt(dx ** 2 + dy ** 2)
            w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0]))

        px[:, ip] = x[:, 0]
        pw[0, ip] = w

    pw = pw / pw.sum()  # normalize

    xEst = px.dot(pw.T)
    PEst = calc_covariance(xEst, px, pw)

    N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
    if N_eff < NTh:
        px, pw = re_sampling(px, pw)
    return xEst, PEst, px, pw


def re_sampling(px, pw):
    """
    low variance re-sampling
    """

    w_cum = np.cumsum(pw)
    base = np.arange(0.0, 1.0, 1/NP)
    re_sample_id = base + np.random.uniform(0, 1/NP)
    indexes = []
    ind = 0
    for ip in range(NP):
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)

    px = px[:, indexes]
    pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

    return px, pw


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eig_val, eig_vec = np.linalg.eig(Pxy)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eig_val[big_ind] or eiq_val[small_ind] were occasionally negative numbers extremely
    # close to 0 (~10^-20), catch these cases and set the respective variable to 0
    try:
        a = math.sqrt(eig_val[big_ind])
    except ValueError:
        a = 0

    try:
        b = math.sqrt(eig_val[small_ind])
    except ValueError:
        b = 0

    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eig_vec[big_ind, 1], eig_vec[big_ind, 0])
    Rot = np.array([[math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]])
    fx = Rot.dot(np.array([[x, y]]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))

    px = np.zeros((4, NP))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight

    # history
    hxEst = xEst
    hxTrue = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input(time)

        xTrue, z, ud = observation(xTrue, u)

        xEst, PEst, px, pw = pf_localization(px, pw, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxTrue = np.hstack((hxTrue, xTrue))

    if show_animation:
        plt.cla()
        plt.plot(np.array(hxTrue[0, :]).flatten(),
                 np.array(hxTrue[1, :]).flatten(), "-b")
        plt.plot(np.array(hxEst[0, :]).flatten(),
                 np.array(hxEst[1, :]).flatten(), "-r")
        #plot_covariance_ellipse(xEst, PEst)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)


if __name__ == '__main__':
    for i in range(10):
        main()
    input()
