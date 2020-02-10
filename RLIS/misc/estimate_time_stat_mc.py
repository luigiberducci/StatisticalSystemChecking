import math
import argparse

def estimate_sra_time(p, epsilon, delta, s):
    g = (4 * (math.e - 2) * math.log(2 / delta)) / (epsilon ** 2)
    g1 = 1 + (1 + epsilon) * g
    exp_sims = math.ceil(g1 / p)
    exp_time = exp_sims / s / 60 / 60 / 24
    return exp_sims, exp_time

def estimate_oaa_time(p, epsilon, delta, s):
    # 1) SRA to estimate mu_z
    g = (4 * (math.e - 2) * math.log(2 / delta)) / (epsilon ** 2)
    g2 = 2 * (1 + math.sqrt(epsilon)) * (1 + 2 * math.sqrt(epsilon)) * (1 + (math.log(3 / 2) / math.log(2 / delta))) * g
    sra_epsilon = min(1 / 2, math.sqrt(epsilon))
    sra_delta = delta / 3
    sra_sims, sra_time = estimate_sra_time(p, sra_epsilon, sra_delta, s)
    oaa_1_sims = math.ceil(g2 * epsilon / p) * 2    # x2 because each iteration 2 simulations
    exp_s = oaa_1_sims * p
    rho_z = max(exp_s / oaa_1_sims, epsilon * p)
    oaa_2_sims = g2 * rho_z / (p ** 2)
    exp_tot_sims = sra_sims + oaa_1_sims + oaa_2_sims
    exp_time = exp_tot_sims / s / 60 / 60 / 24
    return exp_tot_sims, exp_time


parser = argparse.ArgumentParser()
parser.add_argument("p", type=float, nargs=1, help="event probability")
parser.add_argument("--e", type=float, nargs="?", default=0.5, help="error margin")
parser.add_argument("--d", type=float, nargs="?", default=0.7, help="confidence")
parser.add_argument("--s", type=int, default=3, nargs="?", help="number of simulations/sec")
args = parser.parse_args()
p, s = args.p[0], args.s
e, d = args.e, args.d
sra_sims, sra_time = estimate_sra_time(p, e, d, s)
oaa_sims, oaa_time = estimate_oaa_time(p, e, d, s)

print("Assuming p={}, s={}".format(p, s))
print("SRA(e={}, d={}) requires {} simulations ({:.2f} days).".format(e, d, sra_sims, sra_time))
print("OAA(e={}, d={}) requires {} simulations ({:.2f} days).".format(e, d, oaa_sims, oaa_time))


