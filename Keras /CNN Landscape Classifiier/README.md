# CNN Landccape Classifier

Convolutional neural network built with Tensorflow/Keras to allowfor landscape classification based on one of three categories (desert, forest, polar). The project is based on an example from the Coursera course: *Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning*.

## Data and model specifications
* Since the dataset images had to be manually obtained, and therefore limited in size, 3 x 70 images were chosen to be of a similar type, and most represntative of the landscape type they tipify. All input images were resized to a 150 x 150 image. The model's accuracy levels can most likely be improved through the use of a larger dataset.
* The validation set consists of 3 x 10 images.
* The model consists of five convolutions and were trained over 30 epochs.

Resources:
* Images sourced from Pixabay: https://pixabay.com

## Model results
The program was run over 10 iterations in order to obtain an average validation accuracy.

| Iteration     | Validation Accuracy    | Training Accuracy on Epoch 30 |
| :-----------: | ----------------------:| ------:|
| 1             |   93.3333              | 0.8286 |
| 2             |   100.0000             | 0.7524 |
| 3             |   93.3333              | 0.8714 |
| 4             |   90.0000              | 0.8667 |
| 5             |   100.0000             | 0.8475 |
| 6             |   96.6667              | 0.8714 |
| 7             |   100.0000             | 0.9048 |
| 8             |   96.6667              | 0.8952 |
| 9             |   90.0000              | 0.7857 |
| 10            |   96.6667              | 0.8762 |

| Average       | Validation Accuracy    | Training Accuracy on Epoch 30 |
| :-----------: | ----------------------:| ------:|
| 1             |   95,6667              | 0.8499 |

The model was trained on a set of 3 x 70 input images of various sizes. Below is a sample of some of these images:
</br>
<p align="center">
  <img src="input/desert_forest.png" width="550px"/>
  <img src="input/polar.png" width="550px"/>
</p>

The images that were correctly classified tend to have strong visual characteristics relating to its type, such as the images below:
</br>
<p align="center">
  <img src="output/desert-sand-dunes-691431__480.jpg.png" width="350px"/>
  <img src="output/desert-tree-64310__480.jpg.png" width="350px"/>
  <img src="output/forest-forest-1946477__480.jpg.png" width="350px"/>
  <img src="output/polar-arctic-415209__480.jpg.png" width="350px"/>
</p>

Images that were most likely to be misclassified are those that are sligtly marginal, for example a forest scene with a lot of orange-brown tones, or a desert scene with a group of cactus trees.
</br>
<p align="center">
  <img src="misclassified/desert-saguaro-cactus-584405__480.png" width="350px"/>
  <img src="misclassified/desert-desert-2631340__480.png" width="350px"/>
  <img src="misclassified/desert-desert-1363152__480.png" width="350px"/>
  <img src="misclassified/forest-wood-3107139__480.png" width="350px"/>
</p>