# machine-learning
A repo that stores all my project files as I learn and perfect machine learning

## Setup

Install Python3 and Pip3

Install Virtualenv

`pip install virtualenv`

Create ml virtualenv

`python -m venv ml`

Activate Virtualenv

`source ml/bin/activate`

Install packages from requirements.txt

`pip install -r requirements.txt`

## Intro

Files are in order 1_, 2_, 3, etc

Explanation of each file can be found below.

## MNIST Data Used

Credit: Yann LeCun (http://yann.lecun.com/exdb/mnist/)

## Reference Blog Used

Credit: Colah (http://colah.github.io/posts/2014-10-Visualizing-MNIST/)

## And of course Tensorflow

Credit: Google AI Team (https://www.tensorflow.org/)

### 1_first_tests.py

This was the first file I worked with/in that starts with importing and basic playgrounding on Tensorflow. There is a lot of print statements to make sense of it all and its a great place to start if you are reading this as an absolute begginer

### 2_mnist_starting_example.py

This file is where I started working with NumPy and the MNIST data sets, they are also located in this repo and if you clone and use the python scripts in this repository it will most likely utilize the ones here in this repo. (Untested). This file was majorly a copy from the Tensorflow docs on MNIST basics. It was there for me to analyze and fiddle with.

###  3_mnist_first_tests.py

Here I started getting my feet wet with the MNIST data. Achiving ~92% accuracy with simple methods like `GradientDescentOptimizer` and `softmax regression`.

### 4_mnist_softmax_regression.py

I started to use Softmax Regression correctly here, with `softmax_cross_entropy_with_logits` but was still using `GradientDescentOptimizer` and only achiving low 90s for accuracy. 

### 5_mnist_multilayer_convolutional_network.py

Now finally I got into the meat of machine learning with the MNIST dataset utilizing far better methods like a `Multilayer Convolutional Network`. These topics and words may seem confusing but the documentation does a good job of explaining them, as well as a blog post by Colah (http://colah.github.io/posts/2014-10-Visualizing-MNIST/) which explains how even visualizing these Tensors and machine learning networks are tricky at best. There is an entire field behind this topic called `dimensionality reduction` and it's very interesting and a good path to follow if you plan on visualizing your Machine Learning programs. With the `Multilayer Convolutional Network` I was able to achive 100% accuracy on 1600-1700 iterations, of course over time this dipped back into the upper 90s.

### 6_openai_cartpole_test.py

Start of the OpenAI gym tests, here I simply have gotten the cartpole running and am planning to write code to complete it.

### 7_openai_flashgames_test.py

Same as 6, here I simply have gotten flash games running and am planning to write code to complete it.
