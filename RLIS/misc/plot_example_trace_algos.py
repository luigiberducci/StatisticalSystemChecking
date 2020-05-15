import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys, numpy as np

model_dir = os.path.join('/home', 'luigi', 'Development', 'StatisticalSystemChecking', 'RLIS')
sys.path.append(model_dir)

from env.EKF import EKFSystem as EKF
from env.TR import TRSystem as TR

def update(i):
    # import ipdb
    # ipdb.set_trace()
    trueline.set_data(truedata[0, :i], truedata[1, :i])
    beliefline.set_data(beliefdata[0, :i], beliefdata[1, :i])
    gps.set_sizes(np.concatenate([np.ones(i) * 10, np.ones(truedata.shape[1]-i) * 0]))
    return trueline, beliefline, gps

j = 0
def update_tr(i):
    global j
    if i==0:
        j=0
    trueline.set_data(truedata[0, :i], truedata[1, :i])
    beliefline.set_data(beliefdata[0, :i], beliefdata[1, :i])
    if i % s.apply_interval == 0:
        lastinputline.set_data(besttrajx[:, j], besttrajy[:, j])
        j = j + 1
    # ax.scatter(besttrajx[:, i], besttrajy[:, i])
    return trueline, beliefline, ax

s = EKF()
s.run_system()
fig, ax = plt.subplots()

truedata = s.hx_true[0:2, :]
beliefdata = s.hx_est[0:2, :]
observations = s.sensors_data

trueline, = ax.plot(truedata[0, :], truedata[1, :], "-g", label="True state")  # plot the true trajectory
beliefline, = ax.plot(beliefdata[0, :], beliefdata[1, :], "-b", label="Belief state")  # plot the belief trajectory
gps = ax.scatter(observations[0, :], observations[1, :], label="GPS", c='red')
ax.legend()
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')

# anim = animation.FuncAnimation(fig, update, frames=np.arange(1, truedata.shape[1]), interval=35)
# anim.save('ekf_trace.gif', dpi=120, writer='imagemagick')
plt.show()

s = TR()
s.run_system()
fig, ax = plt.subplots()

truedata = s.hx_true[0:2, :]
beliefdata = s.hx_est[0:2, :]
besttrajx= s.hx_xpred
besttrajy= s.hx_ypred

trueline, = ax.plot(truedata[0, :], truedata[1, :], "-g", label="True state")  # plot the true trajectory
beliefline, = ax.plot(beliefdata[0, :], beliefdata[1, :], "-b", label="Belief state")  # plot the belief trajectory
lastinputline, = ax.plot(besttrajx[0], besttrajy[1], "-r", label="Input")  # plot the belief trajectory
ax.legend()
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_xlim((0, 10))
ax.set_ylim((-4, 2))

# Plot obstacles
for x, y in s.obstacle_coord:
    ax.scatter(x, y, marker='+')
    ax.add_artist(plt.Circle((x, y), color='black', fill=True, radius=s.obstacle_radius))
    ax.add_artist(plt.Circle((x, y), color='r', fill=False, radius=s.obstacle_radius + s.car_radius))
    ax.add_artist(plt.Circle((x, y), color='b', fill=False,
                    radius=s.obstacle_radius + s.car_radius + s.safe_margin))
# Plot goal
x = s.x_goal[0]
y = s.x_goal[1]
ax.add_artist(plt.Circle((x, y), color='g', fill=True, radius=s.goal_radius))
ax.add_artist(plt.Circle((x, y), color='y', fill=False, radius=s.goal_radius + s.car_radius))


# anim = animation.FuncAnimation(fig, update_tr, frames=np.arange(0, truedata.shape[1]), interval=35)
# anim.save('tr_trace.gif', dpi=120, writer='imagemagick')
plt.show()
