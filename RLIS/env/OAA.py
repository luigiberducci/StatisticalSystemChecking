import math
from TR import TRSystem


def sra(sys, epsilon, delta):
    assert 0 < epsilon <= 1
    assert 0 < delta <= 1
    g = (4 * (math.e - 2) * math.log(2 / delta)) / (epsilon ** 2)
    g1 = 1 + (1 + epsilon) * g
    n, s = 0, 0
    print("[Info] SRA (e={}, d={}) until >{} counter-examples...".format(epsilon, delta, g1))
    while s <= g1:
        # Simulate
        sys.reset_init_state()
        sys.run_system()
        z = 1 if sys.reward > 1 else 0
        s += z
        n += 1
    return s / n


def oaa(sys, epsilon, delta):
    assert 0 < epsilon <= 1
    assert 0 < delta <= 1
    # 1) SRA to estimate mu_z
    e = math.e
    g = (4 * (e - 2) * math.log(2 / delta)) / (epsilon ** 2)
    g2 = 2 * (1 + math.sqrt(epsilon)) * (1 + 2 * math.sqrt(epsilon)) * (1 + (math.log(3 / 2) / math.log(2 / delta))) * g

    sra_epsilon = min(1 / 2, math.sqrt(epsilon))
    sra_delta = delta / 3
    # mu_z = sra(sys, sra_epsilon, sra_delta)
    mu_z = 0.0001
    print("[Info] SRA returned {}".format(mu_z))

    # 2) use mu_z estimation to estimate rho_z
    n = math.ceil(g2 * epsilon / mu_z) * 0
    s = 0
    print("[Info] OAA 1 - {} samples to estimate rho.".format(n))
    for i in range(n):
        # Simulate once
        sys.reset_init_state()
        sys.run_system()
        z1 = 1 if sys.reward > 1 else 0
        # Simulate again
        sys.reset_init_state()
        sys.run_system()
        z2 = 1 if sys.reward > 1 else 0
        # Estimate
        s += ((z1 - z2) ** 2) / 2
    #rho_z = max(s / n, epsilon * mu_z)
    rho_z = 0.00009986
    print("[Info] OAA 1 End. Rho = {}".format(rho_z))

    # 3) use rho_z to obtain the final approximation mu
    n = g2 * rho_z / (mu_z ** 2)
    s = 0
    print("[Info] OAA 2 - {} samples to estimate mu.".format(n))
    for i in range(n):
        # Simulate once
        sys.reset_init_state()
        sys.run_system()
        z = 1 if sys.reward > 1 else 0
        # Estimate
        s += z
    mu = s / n
    print("[Info] OAA 2 End. mu = {}".format(mu))
    return mu


def main():
    sys = TRSystem()
    epsilon = 0.9  # error in relative error
    delta = 0.5
    oaa(sys, epsilon, delta)


if __name__ == "__main__":
    main()
