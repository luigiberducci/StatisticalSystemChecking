import argparse

class Parser:
    def __init__(self):
        self.parser = self.define_parser()

    def define_parser(self):
        problems = ['EKF', 'SR']
        searches = ['IS', 'MC']
        parser = argparse.ArgumentParser()
        parser.add_argument('--problem',  default='EKF', nargs=1, help='Problem name', choices=problems)
        parser.add_argument('--search',   default='IS',  nargs=1, help='Search method', choices=searches)
        parser.add_argument('--simsteps', default=10000, nargs='?', type=int,
                                          help='Total number of simulation steps (divided into episodes)')
        parser.add_argument('--outdir',   default='out', nargs='?', help='Output directory for results')
        parser.add_argument('--interval', default=10000, nargs='?', type=int,
                            help='Interval (in # simsteps) in which save model')
        parser.add_argument('--envparams', default=[], nargs='*', type=str,
                            help='Environment parameters as list of strings (e.g. P in SR)')
        parser.add_argument('--agentparams', default=[], nargs='*', type=str,
                            help='Agent parameters as list of strings (mem_limit, mem_window_len, warmup_steps, gamma, optimizer, metrics)')
        parser.add_argument('--splitparams', default=[], nargs='*', type=str,
                            help='Importance Splitting parameters as list of strings (num_particles, k_particles, warmup_steps, delta_level)')
        return parser

    def parse_args(self):
        args = self.parser.parse_args()
        problem_params = self.parse_problem_params(args)
        env_params = self.parse_env_params(problem_params['problem_name'], args)
        agent_params = self.parse_agent_params(args)
        split_params = self.parse_split_params(args)
        return problem_params, env_params, agent_params, split_params

    def parse_problem_params(self, args):
        prob_params = dict()
        prob_params['problem_name'] = args.problem[0]  # because it is a list
        prob_params['simsteps'] = args.simsteps
        prob_params['run_is_flag'] = args.search[0] == 'IS'  # run impsplit (is) or mc
        prob_params['outdir'] = args.outdir  # output directory for results
        prob_params['save_interval'] = args.interval  # save interval for models
        return prob_params

    def parse_split_params(self, args):
        # RULE: if 0 params -> default, otherwise always 4 agent parameters:
        # Order is IMPORTANT: num_particles, k_particles, warmup_steps, delta_level
        # If you want ot use a default value, set that parameter as -1
        assert len(args.splitparams) == 0 or len(args.splitparams) == 4
        split_params = dict()
        if len(args.splitparams)>0:
            if not args.splitparams[0] == '-1':  # num_particles
                split_params['num_particles'] = int(args.splitparams[0])
            if not args.splitparams[1] == '-1':  # k particles
                split_params['k_particles'] = int(args.splitparams[1])
            if not args.splitparams[2] == '-1':  # warmup steps
                split_params['warmup_steps'] = int(args.splitparams[2])
            if not args.splitparams[3] == '-1':  # delta level
                split_params['delta_level'] = str(args.splitparams[4])
        return split_params


    def parse_agent_params(self, args):
        # RULE: if 0 params -> default, otherwise always 6 agent parameters:
        # Order is IMPORTANT: mem_limit, mem_window_len, warmup_steps, gamma, optimizer, metrics
        # If you want ot use a default value, set that parameter as -1
        # NOTE: from the 6-th parameter (including), I consider all of them in metrics list
        assert len(args.agentparams) == 0 or len(args.agentparams) >= 6
        agent_params = dict()
        if len(args.agentparams)>0:
            if not args.agentparams[0] == '-1':  # mem limit
                agent_params['mem_limit'] = int(args.agentparams[0])
            if not args.agentparams[1] == '-1':  # mem window len
                agent_params['mem_window_len'] = int(args.agentparams[1])
            if not args.agentparams[2] == '-1':  # warmup steps
                agent_params['warmup_steps'] = int(args.agentparams[2])
            if not args.agentparams[3] == '-1':  # gamma
                agent_params['gamma'] = float(args.agentparams[3])
            if not args.agentparams[4] == '-1':  # optimizer
                agent_params['optimizer'] = str(args.agentparams[4])
            if not args.agentparams[5] == '-1':  # metrics
                # note: metrics must be the last because is a list of values
                agent_params['metrics'] = args.agentparams[5:]
        return agent_params

    def parse_env_params(self, problem_name, args):
        # Compute the num of episodes according to the problem
        env_params = dict()
        try:
            if problem_name == 'EKF':
                episode_step = 100
            elif problem_name == 'SR':
                episode_step = 10
                env_params['P'] = float(args.envparams[0]) if len(args.envparams) > 0 else 0.5  # in SR, P is P(x_i+1 | x_i)
            else:
                raise ValueError("problem name {} is not defined".format(problem_name))
        except:
            raise ValueError("envparams are not valid")

        num_episodes = args.simsteps // episode_step
        env_params['num_episodes'] = num_episodes
        env_params['episode_steps'] = episode_step
        return env_params