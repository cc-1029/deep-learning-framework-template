from base import BaseTrainer


class TfTrainer(BaseTrainer):
    """Base trainer class for TensorFlow

    """

    def train(self, use_custom=True):
        if use_custom:
            for epoch in range(1, self.num_epochs + 1):
                for step, (x, y) in enumerate(self.train_dataloader):
                    l = self.training_step(x, y, training=True)
                pass
        else:
            # Use tf.keras way to train model
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=self.metrics)
            self.model.fit(self.train_dataloader,
                           verbose=2,
                           epochs=self.num_epochs,
                           validation_data=self.eval_dataloader)
            self.model.save(self.saved_model_path)
