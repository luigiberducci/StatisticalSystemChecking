import os
from rl.callbacks import ModelIntervalCheckpoint
from ImportanceSplittingCallback import ImportanceSplittingCallback as IS

class ISFactory:
    def __init__(self, env, agent, outdir, save_interval=None, run_is_flag=True, params=dict()):
        self.manage_params(params)
        assert 'num_particles' in self.split_params.keys()
        assert 'k_particles' in self.split_params.keys()
        assert 'warmup_steps' in self.split_params.keys()
        assert 'delta_level' in self.split_params.keys()
        self.isplit = IS(env, agent, self.split_params['num_particles'], self.split_params['k_particles'],
                         self.split_params['warmup_steps'], self.split_params['delta_level'], run_is_flag, outdir)
        self.save_model_manager = self.init_save_model_manager(outdir, save_interval)

    def get_isplit(self):
        return self.isplit

    def get_callback_list(self):
        #`IS` and `save_model_manager` are implemented as Keras Callbacks
        callbacks_list = [self.isplit]
        if self.save_model_manager is not None:
            callbacks_list.append(self.save_model_manager)
        return callbacks_list

    def init_save_model_manager(self, outdir, interval):
        if interval is not None:
            model_dir = os.path.join(outdir, "models")
            if not os.path.exists(model_dir):  # Create if necessary
                os.makedirs(model_dir, exist_ok=True)
            fp = os.path.join(model_dir, 'weights.{step:02d}.hdf5')
            return ModelIntervalCheckpoint(filepath=fp, interval=interval, verbose=1)
        return None

    def manage_params(self, params):
        self.set_default_params()
        if len(params.keys()) > 0:
            self.update_params(params)

    def set_default_params(self):
        self.split_params = dict()
        self.split_params['num_particles'] = 100
        self.split_params['k_particles'] = 10
        self.split_params['warmup_steps'] = 50
        self.split_params['delta_level'] = 0

    def update_params(self, params):
        param_names = ['num_particles', 'k_particles', 'warmup_steps', 'delta_level']
        for name in param_names:
            if name in params.keys():
                self.split_params[name] = params[name]

    def print_config(self):
        self.isplit.print_info_config()
