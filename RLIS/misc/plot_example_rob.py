import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

t = np.linspace(0, 10, 100)
x = t + 0.1 * (np.random.rand(100)-0.5)
x2 = t + 0.1* (np.random.rand(100)-0.5)
y = 0.04 * t**2
yy = np.sin(x) + y
yy2 = np.sin(x) - y

xg, yg = 5, 0
thr = 0.25
eps = 0.001

omega = np.array((x, yy))
omega2 = np.array((x2, yy2))
refx = np.ones((100)) * xg
refy = np.ones((100)) * yg
ref = np.array((refx, refy))
r = np.linalg.norm(omega-ref, axis=0) - thr - eps
r2 = np.linalg.norm(omega2-ref, axis=0) - thr - eps

def minrob(array):
    out = np.ones(array.shape) * array[-1]
    for i in range(array.shape[0]-2, -1, -1):
        out[i] = min(out[i + 1], array[i])
    return out

e1 = np.exp(-minrob(r))
e2 = np.exp(-minrob(r2))

fig, (ax1, ax2) = plt.subplots(2, 1)
# fig, (ax2, ax3) = plt.subplots(2, 1)

line1, = ax1.plot(x, yy, label="Execution Trace 1")
line2, = ax1.plot(x2, yy2, label="Execution Trace 2")
ax1.scatter(xg, yg, marker='*', color='r')
ax1.add_artist(plt.Circle((xg, yg), thr, fill=False, color='r', alpha=0.5, label="Collision Margin"))
ax1.set_xlabel("X Position")
ax1.set_ylabel("Y Position")
ax1.legend()

rob1, = ax2.plot(t, r, label="Execution Trace 1")
rob2, = ax2.plot(t, r2, label="Execution Trace 2")
ax2.axhline(y=0, color='k')
ax2.set_xlabel("Time")
ax2.set_ylabel("Robustness")

# rew1, = ax3.plot(t, e1, label="Execution Trace 1")
# rew2, = ax3.plot(t, e2, label="Execution Trace 2")
# ax3.axhline(y=1, color='k')
# ax3.set_xlabel("Time")
# ax3.set_ylabel("Reward")
# ax3.set_yscale('log')
# ax3.legend(loc='center right')

def update(i):
    # line1.set_data(x[:i], yy[:i])
    # line2.set_data(x2[:i], yy2[:i])
    rob1.set_data(t[i:], r[i:])
    rob2.set_data(t[i:], r2[i:])
    rew1.set_data(t[i:], e1[i:])
    rew2.set_data(t[i:], e2[i:])
    # return line1, line2, rob1, rob2
    return rob1, rob2, rew1, rew2

anim = FuncAnimation(fig, update, frames=np.arange(100, 0, -1), interval=50)
anim.save('backward_reward_animation.gif', dpi=120, writer='imagemagick')
plt.show()
