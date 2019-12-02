from gym.envs.registration import register

register(
    id='succruns-v1',
    entry_point='gym_success_runs.envs:SuccessRunsEnv',
)
