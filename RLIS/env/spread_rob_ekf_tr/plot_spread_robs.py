import matplotlib.pyplot as plt
import numpy as np

ekf_robs = []
tr_robs = []

N = 1000
with open(str(N) + "ekf.txt", "r") as f:
    for l in f.readlines()[:-2]:
        v = l.split(":")[2]
        ekf_robs.append(float(v))
with open(str(N) + "tr.txt", "r") as f:
    for l in f.readlines()[:-2]:
        v = l.split(":")[2]
        tr_robs.append(float(v))
x = np.arange(0, len(ekf_robs))

plt.scatter(x, ekf_robs, label="EKF")
plt.scatter(x, tr_robs, label="TR")
plt.legend()
plt.xlabel("random samples")
plt.ylabel("trace robustness")
plt.show()
