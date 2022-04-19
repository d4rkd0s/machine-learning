# Import TensorFlow
import tensorflow as tf
print(tf.__version__) # find the version number (should be 2.x+)

# Creating tensors with tf.constant

# Create a scalar tensor 
scalar = tf.constant(1337)
print(scalar)

# Create a vector tensor
vector = tf.constant([1, 2, 3, 4, 5])
print(vector)

# Create a matrix tensor
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
