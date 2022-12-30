import tensorflow as tf


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
