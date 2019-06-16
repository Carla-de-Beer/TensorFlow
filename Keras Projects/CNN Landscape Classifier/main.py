# Carla de Beer
# June 2019
# Convolutional neural network built with Tensorflow/Keras to allow
# for landscape classification based on one of three categories (desert, forest, polar).
# The project is based on an example from the Coursera course:
# "Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning".
# Images sourced from Pixabay (https://pixabay.com).

from DataSet import DataSet
from CNN import CNN
from Classifier import Classifier

# Display datasets
DataSet.load_data()

# Define the model
model = CNN.build_model()

# Train the model
unseen = CNN.compile_and_train_model(model)

# Classify through prediction
Classifier.classify_unseen_data(model, unseen)
