import tensorflow as tf

import base_tf


class TfMnistTrainer(base_tf.TfTrainer):
    def _train_step(self, train_data):
        # 定义如何取数据
        inputs, labels = train_data
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            loss = self.loss(labels, outputs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss


if __name__ == '__main__':
    trainer = TfMnistTrainer()