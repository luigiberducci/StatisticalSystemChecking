f = "offline_SR.log"
f = "offline_EKF.log"
infos = []
results = []
details = []
with open(f, "r") as ff:
    for line in ff.readlines():
        if "Info" in line:
            infos.append(line)
        if "Result" in line:
            results.append(line)
        if "Details" in line:
            line = line.split(":")[1].replace("[", "").replace("]", "")
            probs = [float(p) for p in line.split(",")]
            details = details + probs

# plot
import matplotlib.pyplot as plt
import numpy as np
import random
import ipdb
ipdb.set_trace()
random.shuffle(details)
details = np.array(details)

ref_value = 0.0000001024
ref_value = 0.0000090960
abs_error = abs(ref_value - details.mean())
rel_error = abs_error / ref_value
prc_error = rel_error * 100
succ_rate = np.count_nonzero(details>0)
print("[Result]")
print("succ rate {} / {}".format(succ_rate, details.shape[0]))
print("mean {}".format(details.mean()))
print("abs error {}".format(abs_error))
print("rel error {}".format(rel_error))
print("prc error {}".format(prc_error))

plt.plot(details, label="splitting estimation")
plt.hlines(y=ref_value, xmin=-1, xmax=details.shape[0]+1, color='g', label="reference")
plt.hlines(y=details.mean(), xmin=-1, xmax=details.shape[0]+1, color='r', label="mean estimation")
plt.xlabel("Importance Splitting executions")
plt.ylabel("probability")
plt.legend()
plt.show()
