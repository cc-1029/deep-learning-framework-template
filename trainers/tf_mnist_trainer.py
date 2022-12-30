import base_tf


class TfMnistTrainer(base_tf.TfTrainer):
    def train(self, use_custom=False):
        super().train(use_custom=use_custom)


if __name__ == '__main__':
    trainer = TfMnistTrainer()