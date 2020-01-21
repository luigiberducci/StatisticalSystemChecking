from TrajectoryRollout import System

sys = System()
for i in range(10000):
    sys.reset_init_state()
    sys.run_system()
    #sys.render()

import ipdb
ipdb.set_trace()