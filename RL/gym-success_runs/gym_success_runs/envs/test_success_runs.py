from SuccessRunsEnv import FSA, SuccessRunsEnv

def test_sys_stepbystep():
    sys = FSA()
    for i in range(15):
        print("x: {}, t: {}".format(sys.x, sys.t))
        sys.step_system()

def test_sys_trace():
    sys = FSA()
    for i in range(10):
        sys.run_system(num_steps=i)
        print("Run System for {} steps -> x: {}, t: {}".format(i, sys.x, sys.t))
        sys.reset_init_state()

def test_env_step():
    env = SuccessRunsEnv()
    for i in range(100):
        for j in range(15):
            env.step(None)
        print("Trace {}:".format(i))
        print(env.get_trace())
        env.reset()

def test_env_render():
    env = SuccessRunsEnv(P=0.8)
    for i in range(100):
        for j in range(15):
            env.step(None)
            print(env.is_done)
        print("Trace {}:".format(i))
        env.render()
        env.reset()

def run_mc_env(num_sims=1000, debug_k=100):
    env = SuccessRunsEnv(P=0.4)
    n_sat = 0
    for i in range(num_sims):
        if i%debug_k==0:
            print("[Info] Simulation # {}".format(i))
        while not env.is_done:
            env.step(None)
        if env.is_current_trace_false():
            n_sat = n_sat + 1
        env.reset()
    print("[Info] MC Simulation of {} traces, falsification occurred {} -> {}/{}={}".format(num_sims, n_sat,
                                                                                            n_sat, num_sims, n_sat/num_sims))
def main():
    run_mc_env(100000, 1000)

main()