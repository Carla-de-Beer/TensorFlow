"""
Carla de Beer
June 2019
Convolutional neural network built with Tensorflow/Keras to allow
for landscape classification based on one of three categories (desert, forest, polar).
The project is based on an example from the Coursera course:
"Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning".
Images sourced from Pixabay (https://pixabay.com).
"""

from cnn import CNN
from classifier import Classifier
from data_set import DataSet

# Display datasets
DataSet.load_data()

# Define the model
MODEL = CNN.build_model()

# Train the model
UNSEEN = CNN.compile_and_train_model(MODEL)

# Classify through prediction
Classifier.classify_unseen_data(MODEL, UNSEEN)
