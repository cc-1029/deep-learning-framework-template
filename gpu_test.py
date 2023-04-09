import tensorflow as tf
import torch

from backend import tf_backend

print('TensorFlow gpu available: ', tf.config.experimental.list_physical_devices("GPU"))
print(80 * "-")
print('PyTorch gpu available: ', torch.cuda.is_available())
print(80 * "-")

print('TensorFlow gpu available: ', tf_backend.num_gpus())