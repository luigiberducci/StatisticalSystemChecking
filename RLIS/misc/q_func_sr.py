import numpy as np
import math
import matplotlib.pyplot as plt

p = 0.2
q = 1 - p
epsilon = 0.001
max_val = 10
Q = np.zeros((max_val+1, max_val+1))
Q0 = np.zeros((max_val+1, max_val+1))
Q1 = np.zeros((max_val+1, max_val+1))
R = np.zeros((max_val+1, max_val+1))

for i in range(max_val, -1, -1):
    r = (max_val - i - epsilon)
    rn = (max_val - i - epsilon)/(2 * 3)
    rn1 = (max_val - i - epsilon)/(2 * 7)
    Q0[i, max_val] = math.exp(-r)
    Q[i, max_val] = math.exp(-rn)
    Q1[i, max_val] = math.exp(-rn1)
    R[i, max_val] = r

for i in range(max_val-1, -1, -1):
    for j in range(max_val-1, i-1, -1):
        Q[i, j] = p * Q[i+1, j+1] + q * Q[i, j+1]
        Q0[i, j] = p * Q0[i+1, j+1] + q * Q0[i, j+1]
        Q1[i, j] = p * Q1[i+1, j+1] + q * Q1[i, j+1]
        R[i, j] = max(max_val - i - epsilon, max_val - j - epsilon)
        print("{} {}".format(i, j))
        print(Q)
plt.subplot(141)
plt.imshow(Q0)
plt.colorbar()
plt.title("Q-Function (No Scaling Factor)")
plt.subplot(142)
plt.imshow(Q)
plt.colorbar()
plt.title("Q-Function (Scaling Factor 3)")
plt.subplot(143)
plt.imshow(Q1)
plt.colorbar()
plt.title("Q-Function (Scaling Factor 7)")
plt.subplot(144)
plt.imshow(R)
plt.colorbar()
plt.title("Robustness")
plt.show()
