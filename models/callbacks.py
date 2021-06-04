import tensorflow as tf
import numpy as np


class CheckpointCallback(tf.keras.callbacks.Callback):
    """Define the CheckpointCallback to save the model"""

    def __init__(self, directory):
        super(CheckpointCallback, self).__init__()
        self.directory = directory
        self.best_val = tf.Variable(np.inf, trainable=False)

    def set_model(self, model):
        self.model = model
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.directory,
                                                  checkpoint_name='model', max_to_keep=1)

    def on_epoch_end(self, epoch, logs=None):
        val = logs['val_loss']
        if val < self.best_val:
            self.best_val = val
            self.manager.save()