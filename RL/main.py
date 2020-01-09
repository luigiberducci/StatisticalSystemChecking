from parser import Parser
from env.env_factory import EnvFactory
from model.model_factory import ModelFactory
from agent.agent_factory import AgentFactory
from isplit.isplit_factory import ISFactory

def main():
    parser = Parser()
    problem_params, env_params, model_params, agent_params, split_params = parser.parse_args()
    problem_name = problem_params['problem_name']
    run_is_flag = problem_params['run_is_flag']
    save_interval = problem_params['save_interval']
    outdir = problem_params['outdir']
    print("Problem name: {}".format(problem_name))
    # Env, Model, Agent
    env = build_env(problem_name, env_params)
    model = build_model(problem_name, env.observation_space.shape, env.action_space.n, model_params)
    agent = build_agent(problem_name, model, agent_params)
    # Search
    split_manager = ISFactory(env, agent, outdir, save_interval, run_is_flag, split_params)
    isplit = split_manager.get_isplit()
    callbacks = split_manager.get_callback_list()
    # Run all
    search_algo = 'Importance Splitting' if run_is_flag else 'Monte Carlo'
    print("[Info] Start {} on {}.".format(search_algo, problem_name))
    agent.fit(env, nb_steps=problem_params['simsteps'],
              callbacks=callbacks, verbose=2, visualize=False)
    print("[Info] End {}. Falsification occurred {} times.".format(search_algo, isplit.falsification_counter))

def build_env(problem_name, env_params):
    # Environment
    env_manager = EnvFactory(problem_name, env_params)
    env = env_manager.get_env()
    env_manager.print_config()
    return env

def build_model(problem_name, observation_shape, num_actions, model_params):
    # Model
    model_manager = ModelFactory(problem_name, observation_shape, num_actions, model_params)
    model = model_manager.get_model()
    return model

def build_agent(problem_name, model, agent_params):
    # Agent
    agent_manager = AgentFactory(problem_name, model, agent_params)
    agent = agent_manager.get_compiled_agent()
    return agent

if __name__=='__main__':
    main()