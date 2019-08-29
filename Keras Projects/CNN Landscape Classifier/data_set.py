"""
Loading, processing and displaying of incoming files.
"""

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from enums import LandscapeType


class DataSet:

    @staticmethod
    def load_data():
        print(LandscapeType.forest.value)

        # Directory with the training desert pictures
        train_desert_dir = os.path.join('image-database/' + LandscapeType.desert.value)

        # Directory with the training fores pictures
        train_forest_dir = os.path.join('image-database/' + LandscapeType.forest.value)

        # Directory with the training polar pictures
        train_polar_dir = os.path.join('image-database/' + LandscapeType.polar.value)

        train_desert_names = os.listdir(train_desert_dir)
        print(LandscapeType.desert.value.capitalize() + ' names')
        print(train_desert_dir[:10])

        train_forest_names = os.listdir(train_forest_dir)
        print(LandscapeType.forest.value.capitalize() + ' names')
        print(train_forest_names[:10])

        train_polar_names = os.listdir(train_polar_dir)
        print(LandscapeType.polar.value.capitalize() + ' names')
        print(train_polar_names[:10])

        print('Total training desert images:', len(os.listdir(train_desert_dir)))
        print('Total training forest images:', len(os.listdir(train_forest_dir)))
        print('Total training polar images:', len(os.listdir(train_polar_dir)))

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
        next_polar_pix = [os.path.join(train_polar_dir, fname)
                          for fname in train_polar_names[pic_index - 8:pic_index]]

        # Display desert and forest images
        for i, img_path in enumerate(next_desert_pix + next_forest_pix):
            # Set up subplot; subplot indices start at 1
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off')  # Don't show axes (or gridlines)

            img = mpimg.imread(img_path)
            plt.imshow(img)

        plt.savefig('input/desert_forest.png')
        plt.show()

        # Display polar images
        for i, img_path in enumerate(next_polar_pix):
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off')

            img = mpimg.imread(img_path)
            plt.imshow(img)

        plt.savefig('input/polar.png')
        plt.show()
