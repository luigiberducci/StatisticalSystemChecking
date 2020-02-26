from CROSS import CrossRoadEnv
from timeit import default_timer as timer

sys = CrossRoadEnv()
sys.reset_init_state()
sys.run_system()
pre = 10
prefix = sys.get_trace()[:, 0:pre]
k = 0
n = 100
start = timer()
for i in range(n):
    sys.reset_init_state()
    # print("[Info] Car1 Configuration: range1: {}, range2: {}, range3: {}".format(sys.car1.v1range, sys.car1.v2range,
    #                                                                              sys.car1.v3range))
    # print("[Info] Car2 Configuration: range1: {}, range2: {}, range3: {}\n".format(sys.car2.v1range, sys.car2.v2range,
    #                                                                              sys.car2.v3range))
    sys.run_system()
    if sys.is_current_trace_false():
        k +=1
        sys.render()
    sys.render()
elapsed = timer() - start
print("[Info] {} falsification over {} runs in {} seconds".format(k, n, elapsed))
