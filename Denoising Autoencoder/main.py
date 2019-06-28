# Importing tensorflow
import tensorflow as tf
# importing the data
from tensorflow.examples.tutorials.mnist import input_data
# Importing some more libraries
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
from pylab import rcParams


# reading the data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images



# deciding how big we want our print out to be
rcParams['figure.figsize'] = 20,20
# looping through the first 10 test images and printing  them out
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i].reshape(28,28),  cmap='Greys')
    plt.axis('off')
plt.show()
# printing out the noisy images
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test_noisy[i].reshape(28,28),  cmap='Greys')
    plt.axis('off')
plt.show()	
