import tensorflow as tf
import torch

print(tf.config.experimental.list_physical_devices("GPU"))
print(f"======")
print(torch.cuda.is_available())