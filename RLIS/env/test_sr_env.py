from SR import SRSystem as SR

sys = SR(p=0.2)
for i in range(10):
    sys.reset_init_state()
    sys.run_system()
    sys.render(title="Reward: {}".format(sys.reward))