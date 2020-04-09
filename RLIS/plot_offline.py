import matplotlib.pyplot as plt
import numpy as np
import random
import ipdb

names = ["SR", "EKF"]
fff = [ ["offline_SR.log", "offline_SR_noRScale.log"],
        ["offline_EKF.log"] ]
refs = [0.0000001024, 0.0000090960]

for i, (name, files, ref) in enumerate(zip(names, fff, refs)):
    plt.subplot(len(fff), 1, i+1)
    for f in files:
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
        random.shuffle(details)
        details = np.array(details)

        abs_error = abs(ref - details.mean())
        rel_error = abs_error / ref
        prc_error = rel_error * 100
        succ_rate = np.count_nonzero(details>0)
        print("[Result] file: {}".format(f))
        print("succ rate {} / {}".format(succ_rate, details.shape[0]))
        print("mean {}".format(details.mean()))
        print("abs error {}".format(abs_error))
        print("rel error {}".format(rel_error))
        print("prc error {}".format(prc_error))

        plt.plot(details, label="{} split estimations".format(name))
        plt.hlines(y=details.mean(), xmin=-1, xmax=details.shape[0]+1, color='r', label="{} mean est.".format(name))
    plt.hlines(y=ref, xmin=-1, xmax=details.shape[0]+1, color='g', label="{} ref.".format(name))
    plt.xlabel("Importance Splitting executions")
    plt.ylabel("probability")
    plt.legend()
plt.show()
