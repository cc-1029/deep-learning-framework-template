import numpy as np
import tensorflow as tf

def get_mnist():
    with np.load('./data/mnist.npz') as data:
        x_train_full = data['x_train']
        y_train_full = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
    x_train_full = x_train_full / 255.0
    return x_train_full, y_train_full


def get_train_dataloader(batch_size=64):
    x_train_full, y_train_full = get_mnist()
    x_train, y_train = x_train_full[5000:], y_train_full[5000:]
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).cache().shuffle(
            buffer_size=x_train.shape[0],
            seed=42).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds


def get_eval_dataloader(batch_size=64):
    x_train_full, y_train_full = get_mnist()
    x_eval, y_eval = x_train_full[:5000], y_train_full[:5000]
    eval_ds = tf.data.Dataset.from_tensor_slices(
        (x_eval, y_eval)).cache().shuffle(buffer_size=x_eval.shape[0],
                                          seed=42).batch(batch_size).prefetch(
                                              tf.data.experimental.AUTOTUNE)
    return eval_ds
