"""
Classifier.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

from enums import LandscapeType
from enums import ImageInfo


class Classifier:

    @staticmethod
    def classify_unseen_data(model, unseen_names):
        count = 0

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
                message = name + ": This is a {} image".format(LandscapeType.desert.value)
                plt.xlabel(name + ' = ' + LandscapeType.desert.value, fontsize=10)
                # plt.show()
                plt.savefig('output/' + name + '.png')
                if LandscapeType.desert.value in name:
                    count = count + 1
                    print(message + '.')
                else:
                    print(message + ' - WRONG.')
            elif classes[0][1] == 1:
                message = name + ": This is a {} image".format(LandscapeType.forest.value)
                plt.xlabel(name + ' = ' + LandscapeType.forest.value, fontsize=10)
                # plt.show()
                plt.savefig('output/' + name + '.png')
                if LandscapeType.forest.value in name:
                    count = count + 1
                    print(message + '.')
                else:
                    print(message + ' - WRONG.')
            elif classes[0][2] == 1:
                message = name + ": This is a {} image".format(LandscapeType.polar.value)
                plt.xlabel(name + ' = ' + LandscapeType.polar.value, fontsize=10)
                # plt.show()
                plt.savefig('output/' + name + '.png')
                if LandscapeType.polar.value in name:
                    count = count + 1
                    print(message + '.')
                else:
                    print(message + ' - WRONG.')

                    classification_accuracy = count / len(unseen_names)
                    print('Validation accuracy on unseen images: {0:.4f}'.format(classification_accuracy * 100))
