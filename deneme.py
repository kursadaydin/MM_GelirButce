import tensorflow as tf
print("Uyumlu GPU Sayısı: ", len(tf.config.list_physical_devices('GPU')))