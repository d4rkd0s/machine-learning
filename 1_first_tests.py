#!/usr/bin/python -tt

print("[Program Started]")

print("[Importing tensorflow as tf]")
import tensorflow as tf

print("[Importing numpy as np]")
import numpy as np

print("[Building computational graph node1 node2]")
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

print("[Printing node1 and node2]")
print(node1, node2)

print("[Building tensorflow Session]")
sess = tf.Session()

print("[Running tensorflow Session with node1 node2]")
print(sess.run([node1, node2]))

print("[Creating node3 by combining tensor node1 and node2]")
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

print("[Creating adder_node to combine later inputed vars]")
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print("[Running session with adder_node a:3 b:4.5]")
print(sess.run(adder_node, {a: 3, b:4.5}))

print("[Running session with adder_node a:[1,3] b:[2,4]]")
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

print("[Creating add_and_triple node from adder_node * 3]")
add_and_triple = adder_node * 3.

print("[Testing add_and_triple node with a:3 b:4.5]")
print(sess.run(add_and_triple, {a: 3, b:4.5}))

print("[Creating linear_model W * x + b]")
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

print("[Initializing variables with sess.run(init)]")
init = tf.global_variables_initializer()
sess.run(init)

print("[Running linear_model with x:[1,2,3,4]]")
print(sess.run(linear_model, {x:[1,2,3,4]}))

print("[Running loss test with x[1,2,3,4] and y:[0,-1,-2,-3]]")
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

print("[Fixing W:-1. and b:1. and rerunning loss test]")
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

print("[Now for machine learning to learn the correct W and b values]")
print("[Creating optimizer with GradientDescentOptimizer(0.01)]")
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

print("[Running session with 1000 iterations]")
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([W, b]))

print("[Evaluating training accuracy]")
# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

print("[Now using tf.contrib.learn custom model]")
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))


print("[Program Complete]")
