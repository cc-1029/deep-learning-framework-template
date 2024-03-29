import tensorflow as tf


class TfMnistModel(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(args.num_labels)

    def call(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)