import tensorflow as tf
import torch

print('TensorFlow gpu available: ', tf.config.experimental.list_physical_devices("GPU"))
print(80 * "-")
print('PyTorch gpu available: ', torch.cuda.is_available())