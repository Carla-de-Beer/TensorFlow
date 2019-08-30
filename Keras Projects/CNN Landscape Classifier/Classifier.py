"""
Classifier operates on unseen data files.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

from enums import ImageInfo
from enums import LandscapeType


class Classifier:
    count = 0

    @staticmethod
    def classify_unseen_data(model, unseen_names):

        for name in unseen_names:
            img_path = './unseen/' + name
            img = tf.keras.preprocessing.image.load_img(img_path,
                                                        target_size=(ImageInfo.size.value, ImageInfo.size.value))

            plt.imshow(img)

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            if classes[0][0] == 1:
                Classifier.process_classification(name, LandscapeType.desert.value,
                                                  LandscapeType.forest.value, LandscapeType.polar.value)
            elif classes[0][1] == 1:
                Classifier.process_classification(name, LandscapeType.forest.value,
                                                  LandscapeType.desert.value, LandscapeType.polar.value)
            elif classes[0][2] == 1:
                Classifier.process_classification(name, LandscapeType.polar.value,
                                                  LandscapeType.desert.value, LandscapeType.forest.value)

                classification_accuracy = Classifier.count / len(unseen_names)
                print('Validation accuracy on unseen images: {0:.4f}'.format(classification_accuracy * 100))

    @staticmethod
    def process_classification(name, landscape_type, alt1, alt2):
        message = name + ": This is a {} image".format(landscape_type)
        plt.xlabel(name + ' = ' + landscape_type, fontsize=10)
        plt.savefig('output/' + name + '.png')
        if landscape_type in name:
            Classifier.count = Classifier.count + 1
            print(message + '.')
        else:
            print(message + ' - WRONG.')
            type_name = ''
            if alt1 in name:
                type_name = alt1
            elif alt2 in name:
                type_name = alt2
            plt.title('INCORRECTLY CLASSIFIED\n {} instead of {}'.format(landscape_type.upper(), type_name.upper()))
            plt.show()
