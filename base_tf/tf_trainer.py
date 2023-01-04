from base import BaseTrainer


class TfTrainer(BaseTrainer):
    """Base trainer class for TensorFlow

    """

    def train(self, use_custom=False):
        if use_custom:
            for epoch in range(1, self.num_epochs + 1):
                for step, train_data in enumerate(self.train_dataloader):
                    loss = self._train_step(train_data)
                    self.logger.info(f'Epoch: {epoch} Step: {step} Loss: {loss}')
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

    def _train_step(self, train_data):
        raise NotImplementedError