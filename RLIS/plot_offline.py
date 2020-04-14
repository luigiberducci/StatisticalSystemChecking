import matplotlib.pyplot as plt
import numpy as np
import random
import ipdb

names = ["SR", "TR"] # "EKF", "TR"]
fff = [ ["offline_SR_noRScale.log", "offline_SR_oldRScale.log", "offline_SR_newRScale.log"],
        ["offline_EKF_noRScale.log", "offline_EKF_oldRScale.log", "offline_EKF_newRScale.log"],
        ["offline_TR_noRScale.log", "offline_TR_oldRScale.log", "offline_TR_newRScale.log"] ]
refs = [0.0000001024, 0.0000090960, 0.00001429]

for i, (name, files, ref) in enumerate(zip(names, fff, refs)):
    plt.subplot(len(names), 1, i+1)
    for j, f in enumerate(files):
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
        all_details = np.array(details)
        details = all_details[np.nonzero(all_details)]

        abs_error = abs(ref - details.mean())
        rel_error = abs_error / ref
        prc_error = rel_error * 100
        print("[Result] file: {}".format(f))
        print("succ rate {} / {}".format(details.shape[0], all_details.shape[0]))
        print("mean {}".format(details.mean()))
        print("abs error {}".format(abs_error))
        print("rel error {}".format(rel_error))
        print("prc error {}".format(prc_error))

        if j==0:
            lbl = "{} - No Scaling".format(name)
            clr = 'blue'
        elif j==1:
            lbl = "{} - Same Scaling".format(name)
            clr = 'red'
        else:
            lbl = "{} - New Scaling".format(name)
            clr = 'orange'
        if j!=1:
            plt.plot(all_details, c=clr)
            plt.hlines(y=details.mean(), xmin=-1, xmax=all_details.shape[0]+1, label=lbl, color=clr)
    plt.hlines(y=ref, xmin=-1, xmax=all_details.shape[0]+1, color='g', label="{} ref.".format(name))
    plt.xlabel("Importance Splitting executions")
    plt.ylabel("probability")
    plt.legend()
    print()
plt.show()
