import gym
import gym_success_runs
import gym_ekf_localization

class EnvFactory:
    def __init__(self, problem_name, params=dict()):
        self.problem_name = problem_name
        self.parse_params(params)     #here env params are defined as attributes
        self.env = self.create_env()

    def get_env(self):
        return self.env

    def print_config(self):
        self.env.print_info_config()

    def create_env(self):
        assert self.problem_name in ['EKF', 'SR']
        if self.problem_name == 'EKF':
            env_name = 'ekf-loc-v1'
            return gym.make(env_name, err_threshold=self.err_threshold)
        elif self.problem_name == 'SR':
            env_name = 'succruns-v1'
            return gym.make(env_name, P=self.P)

    def parse_params(self, params):
        if self.problem_name == 'EKF':
            self.episode_steps = params['episode_steps']
            self.err_threshold= params['err_threshold']
        elif self.problem_name == 'SR':
            self.episode_steps = params['episode_steps']
            self.P = params['P']
        else:
            raise ValueError("problem name {} is not defined".format(self.problem_name))
