import os

import tensorflow as tf

from Enums import ImageInfo
from Enums import NetworkParams


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > NetworkParams.accuracy.value):
            print("\nReached {}% accuracy. Ending training.".format(NetworkParams.accuracy.value * 100))
            self.model.stop_training = True


class CNN:

    @staticmethod
    def build_model():
        return tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color

            # The first convolution
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                   input_shape=(ImageInfo.size.value, ImageInfo.size.value, 3)),
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

            # 3 output neurons
            tf.keras.layers.Dense(3, activation='softmax')
        ])

    @staticmethod
    def compile_and_train_model(model):
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                      metrics=['accuracy'])

        # All images will be rescaled by 1./255
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

        # Flow training images in batches of 128 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
            'image-database/',
            target_size=(ImageInfo.size.value, ImageInfo.size.value),  # All images will be re-sized to 150x150
            batch_size=128,
            class_mode='categorical')

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=8,
            epochs=NetworkParams.num_epochs.value,
            verbose=1,
            callbacks=[myCallback()])

        unseen_dir = os.path.join('unseen')
        unseen_names = os.listdir(unseen_dir)
        unseen_names.remove('.DS_Store')  # In case of a macOS directory
        unseen_names.sort()

        return unseen_names
