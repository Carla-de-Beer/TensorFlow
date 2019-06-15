# Carla de Beer
# June 2019
# Tensorflow / Keras binary landscape classifier that allows an input image to be classified as either a desert or forest image.
# Based on the example from the Coursera course: "Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning".
# Images sourced from  (https://pixabay.com).

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

NUM_EPOCHS = 30
ACCURACY = 0.98


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > ACCURACY):
            print("\nReached {}% accuracy. Ending training.".format(ACCURACY * 100))
            self.model.stop_training = True


# Directory with our training desert pictures
train_desert_dir = os.path.join('desert-or-forest/deserts')

# Directory with our training forests pictures
train_forest_dir = os.path.join('desert-or-forest/forests')

train_desert_names = os.listdir(train_desert_dir)
print('Desert names')
print(train_desert_dir[:10])

train_forest_names = os.listdir(train_forest_dir)
print('Forest names')
print(train_forest_names[:10])

print('Total training desert images:', len(os.listdir(train_desert_dir)))
print('Total training forest images:', len(os.listdir(train_forest_dir)))

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_desert_pix = [os.path.join(train_desert_dir, fname)
                   for fname in train_desert_names[pic_index - 8:pic_index]]
next_forest_pix = [os.path.join(train_forest_dir, fname)
                   for fname in train_forest_names[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_desert_pix + next_forest_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x200 with 3 bytes color

    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),

    # 2048 neuron hidden layer
    tf.keras.layers.Dense(2048, activation='relu'),

    # Only 1 output neuron.
    # It will contain a value from 0-1 where 0 for 1 class ('deserts') and 1 for the other ('forests').
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'desert-or-forest/',  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=NUM_EPOCHS,
    verbose=1,
    callbacks=[myCallback()])

unseen_dir = os.path.join('unseen')
unseen_names = os.listdir(unseen_dir)
unseen_names.remove('.DS_Store')  # In case of a macOS directory
unseen_names.sort()

for name in unseen_names:
    img_path = './unseen/' + name
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))

    imgplot = plt.imshow(img)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    if classes[0][0] > 0.5:
        print(name + ": I am {}% certain that this is a forest image.".format(classes[0][0] * 100))
        plt.xlabel(name + ' = forest', fontsize=10)
        plt.show()
    else:
        print(name + ": I am {}% certain that this is a desert image.".format(classes[0][0] * 100))
        plt.xlabel(name + ' = desert', fontsize=10)
        plt.show()
