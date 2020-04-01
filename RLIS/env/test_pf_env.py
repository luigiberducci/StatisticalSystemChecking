from PF import PFSystem
from timeit import default_timer as timer

template_print = "[system] complete run: episode: {}, end condition: {}, reward: {}"
template_plot = "Episode {} - Robustness: {:4.3f}, Reward: {:4.3f}"

sys = PFSystem(num_particles=100)
num_sims = 100
render_flag = True
error_counter = 0

start = timer()
for i in range(num_sims):
    end_condition = "SAFE"
    sys.reset_init_state()
    sys.run_system()
    log_info = template_plot.format(i, sys.robustness, sys.reward)
    if sys.reward > 1:
        end_condition = "ERROR"
        error_counter += 1
    if render_flag:
        sys.render(title=log_info)
    print(template_print.format(i+1, end_condition, sys.reward))
elapsed_time = timer() - start
print("[Result] {} Simulations (plot={}) in {} seconds. Num errors: {}".format(num_sims, render_flag,
                                                                                   elapsed_time, error_counter))
