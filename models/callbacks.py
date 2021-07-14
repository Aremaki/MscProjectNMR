import tensorflow as tf
import numpy as np
import time
import joblib
import os
import matplotlib.pyplot as plt

class CheckpointCallback(tf.keras.callbacks.Callback):
    """Define the CheckpointCallback to save the model"""

    def __init__(self, directory):
        super(CheckpointCallback, self).__init__()
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.time()
        self.directory = directory
        self.best_val = tf.Variable(np.inf, trainable=False)

    def set_model(self, model):
        self.model = model
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.directory,
                                                  checkpoint_name='model', max_to_keep=1)

    def on_epoch_end(self, epoch, logs=None):
        self.times.append((epoch, time.time() - self.timetaken))
        val = logs['val_loss']
        if val < self.best_val:
            self.best_val = val
            self.manager.save()
            
    def on_train_end(self,logs = {}):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.show()
        joblib.dump(self.times, os.path.join(self.directory, 'recorded_times'))