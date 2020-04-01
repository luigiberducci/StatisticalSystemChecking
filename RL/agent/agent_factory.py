from rl.memory import SequentialMemory
from agent.REINFORCE import REINFORCE

class AgentFactory:
    def __init__(self, problem_name="", model=None, agent_params=dict()):
        assert model is not None
        assert problem_name!="" or len(agent_params.keys())>0
        self.manage_params(problem_name, agent_params)
        assert 'mem_limit' in self.agent_params.keys()
        assert 'mem_window_len' in self.agent_params.keys()
        assert 'gamma' in self.agent_params.keys()
        assert 'batch_size' in self.agent_params.keys()
        assert 'warmup_steps' in self.agent_params.keys()
        # Replay Memory
        replay_memory = SequentialMemory(limit=self.agent_params['mem_limit'],
                                         window_length=self.agent_params['mem_window_len'])
        # Build Agent
        self.agent = REINFORCE(model, replay_memory, self.agent_params['gamma'],
                               batch_size=self.agent_params['batch_size'], nb_steps_warmup=self.agent_params['warmup_steps'])

    def manage_params(self, problem_name, agent_params):
        self.set_default_params(problem_name)
        if len(agent_params.keys()) > 0:
            self.update_params(agent_params)

    def set_default_params(self, problem_name):
        # problem name might be "" if the parameters are all defined by user input
        if problem_name == 'EKF':
            self.agent_params = self.get_default_params_EKF()
        elif problem_name == 'SR':
            self.agent_params = self.get_default_params_SR()
        else:
            self.agent_params = dict()

    def get_default_params_EKF(self):
        params = dict()
        params['mem_limit'] = 1000
        params['mem_window_len'] = 1
        params['gamma'] = 1
        params['warmup_steps'] = 1000
        params['batch_size'] = 1
        params['optimizer'] = 'sgd'
        params['metrics'] = ['mae']
        return params

    def get_default_params_SR(self):
        params = dict()
        params['mem_limit'] = 500
        params['mem_window_len'] = 1
        params['gamma'] = 1
        params['warmup_steps'] = params['mem_limit']
        params['batch_size'] = 1
        params['hidden_activation'] = 'relu'
        params['last_activation'] = 'linear'
        params['optimizer'] = 'sgd'
        params['metrics'] = ['mae']
        return params

    def update_params(self, params):
        param_names = ['mem_limit', 'mem_window_len', 'gamma',
                       'warmup_steps', 'batch_size', 'hidden_activation', 'last_activation', 'optimizer', 'metrics']
        for name in param_names:
            if name in params.keys():
                self.agent_params[name] = params[name]

    def get_compiled_agent(self):
        assert 'optimizer' in self.agent_params.keys()
        assert 'metrics' in self.agent_params.keys()
        self.agent.compile(optimizer_name=self.agent_params['optimizer'],
                           metrics=self.agent_params['metrics'])
        return self.agent