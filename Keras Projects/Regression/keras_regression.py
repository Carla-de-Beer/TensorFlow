# Carla de Beer
# Created: May 2019
# Basic keras regression example, taken from the Coursera course: "Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning".

import tensorflow as tf
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

model.fit(xs, ys, epochs=1000)
print(model.predict([7.0]) * 100)

plt.title("Regression graph: house size relative to price")
plt.xlabel('Number of bedrooms')
plt.ylabel('House prices')
plt.plot(xs, ys)
plt.show()
