from rl.callbacks import Callback, ModelIntervalCheckpoint
import tensorflow as tf

class ModelIntervalCompleteCheckpoint(Callback):
    def __init__(self, filepath, interval, verbose=0):
        super(ModelIntervalCompleteCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(
                self.total_steps, filepath))
        import ipdb
        ipdb.set_trace()

        #NOTE: NOT WORKING BECAUSE REQUIRE TF EAGER EXECUTION
        self.model.model.save(filepath, overwrite=True)
