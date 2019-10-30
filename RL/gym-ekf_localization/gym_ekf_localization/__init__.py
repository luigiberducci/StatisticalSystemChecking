from gym.envs.registration import register

register(
    id='ekf-loc-v0',
    entry_point='gym_ekf_localization.envs:EKFLocEnv',
)
