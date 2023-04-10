import random

import numpy as np
import tensorflow as tf

from base import base_trainer

class TfTrainer(base_trainer.BaseTrainer):
    """Base trainers class for TensorFlow

    """
    def init_optimizer(self, parser):
        self.optimizer = parser.init_obj('optimizer', learning_rate=self.learning_rate)

    def train(self):
        if self.use_custom:
            for epoch in range(1, self.num_epochs + 1):
                for step, train_data in enumerate(self.train_dataloader):
                    loss = self._train_step(train_data)
                    # self.logger.info(f'Epoch: {epoch} Step: {step} Loss: {loss}')
        else:
            # Use tf.keras way to train models
            if hasattr(self, 'metrics'):
                self.model.compile(loss=self.loss,
                                   optimizer=self.optimizer,
                                   metrics=self.metrics)
            else:
                self.model.compile(loss=self.loss,
                                   optimizer=self.optimizer)
            self.model.fit(self.train_dataloader,
                           verbose=2,
                           epochs=self.num_epochs,
                           validation_data=self.val_dataloader)
            # self.model.save(self.saved_model_path)

    def _train_step(self, train_data):
        raise NotImplementedError

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def cpu():
    return tf.device('/CPU:0')


def gpu(i=0):
    return tf.device(f'/GPU:{i}')


def num_gpus():
    return len(tf.config.experimental.list_physical_devices('GPU'))


def try_gpu(i=0):
    if num_gpus() > i:
        return gpu(i)
    return cpu()


def try_all_gpus():
    return [gpu(i) for i in range(num_gpus())]
