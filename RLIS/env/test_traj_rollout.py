from TR import TRSystem as System
from timeit import default_timer as timer

template_run = "[system] complete run: episode: {}, end condition: {}, reward: {}"
template_log = "[log] episode: {}/{}, collisions: {}, elapsed time: {}"

sys = System()
start = timer()
collision_counter = 0
num_sims = 100
print_interval = 1000
render_flag = False
print_flag = False # disable printing (but collision and interval log)
sum_rob = 0
for i in range(num_sims):
    sys.reset_init_state()
    sys.run_system()
    reward = sys.reward
    condition = "GOAL"
    if reward > 1:
        condition = "COLLISION"
        collision_counter += 1
        print(template_run.format(i + 1, condition, reward))
        sys.render()
    if render_flag:
        sys.render("{}".format(i))
    if print_flag or i % print_interval == 0:
        elapsed_time = timer() - start
        #print(template_log.format(i + 1, num_sims, collision_counter, elapsed_time))
    print("Episode: {}, Rob: {}".format(i+1, sys.robustness))
    sum_rob += sys.robustness
avg_rob = sum_rob / num_sims
print("Rob: sum: {}, num samples: {} -> avg rob: {}".format(sum_rob, num_sims, avg_rob))
elapsed_time = timer() - start
print("[Result] {} Simulations (plot={}) in {} seconds. Num collisions: {}".format(num_sims, render_flag, elapsed_time, collision_counter))