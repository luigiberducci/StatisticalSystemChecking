from gym.envs.registration import register

register(
    id='fastslam-v0',
    entry_point='gym_ekf_localization.envs:FastSlamEnv',
)
