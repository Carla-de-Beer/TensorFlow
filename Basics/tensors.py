# Carla de Beer
# Created: April 2019
# The basics on tensors (tf.Variable, tf.constant, tf.placeholder)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# tf.Variable: represents a shared, persistent state
my_variable = tf.get_variable('my_variable', [1, 2, 3])
my_int_variable = tf.get_variable('my_int_variable', [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)
other_variable = tf.get_variable('other_variable', dtype=tf.int32, initializer=tf.constant([23, 42]))

# Rank 0
mammal = tf.Variable('Elephant', tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

# Rank 1
mystr = tf.Variable(['Hello'], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

# Rank 2
mymat = tf.Variable([[7], [11]], tf.int32)
myxor = tf.Variable([[False, True], [True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
rank_of_squares = tf.rank(squarish_squares)

# tf.constant
# Rank 0
tensor0D = tf.constant(4)

# Rank 1
tensor1D = tf.constant([4])
tensor1DVector = tf.constant([4, 2], tf.int16, [2], 'tensor1DVector', True)

# Rank 2
constant = tf.constant(17, dtype=tf.int32, shape=[4, 3], name='constant', verify_shape=False)
tensor2D = tf.constant([[1, 2], [5, 7]])
tensor2DAgain = tf.constant([[1, 2], [5, 7], [9, 11]])

# Rank 3
tensor3D = tf.constant(5, dtype=tf.int64, shape=[2, 4, 3], name='constant', verify_shape=False)

# Special tensor functions
zero_tensor1 = tf.zeros([3], dtype=tf.int32)
zero_tensor2 = tf.zeros([3])
one_tensor = tf.ones([4, 3])
fill_tensor = tf.fill([1, 2, 3], 81.0)

# range
range_tensor = tf.range(10, 50, delta=5)

# linear
lin_tensor = tf.lin_space(5., 9., 4)

# random
rnd_normal = tf.random_normal([10], dtype=tf.float64)
rnd_uniform = tf.random_uniform([10], minval=-1, maxval=1, dtype=tf.float32)

# reshaped
vec1 = tf.constant([1., 2., 3., 4.])
reshaped = tf.reshape(vec1, [2, 2])

# reversed
vec2 = tf.constant([[1., 2., 3.], [4., 5., 6.]])
reverseAxis0 = tf.reverse(vec2, [0])
reverseAxis1 = tf.reverse(vec2, [1])
reverseAxis2 = tf.reverse(vec2, [0, 1])

# stacked
t4 = tf.constant([1., 2.])
t5 = tf.constant([3., 4.])
t6 = tf.constant([5., 6.])
stack1 = tf.stack([t4, t5, t6], axis=0)
stack2 = tf.stack([t4, t5, t6], axis=1)

# String join
msg = tf.string_join(['Hello', ' ', 'TensorFlow!'])

# tf.placeholder
p = tf.placeholder(tf.float64)
t = p + [1.2, 4.4]

# Create a tensor collection
tf.add_to_collection("my_collection_name", my_variable)
tf.add_to_collection("my_collection_name", constant)

# Launch session
# Print with either eval() or sess.run()
# Tensor.eval returns a numpy array
with tf.Session() as sess:

    # tf.Variable
    print('\n-------------------------------------------------')
    print('tf.Variable -------------------------------------')
    print('-------------------------------------------------')

    # Initialize all trainable variables in one go
    sess.run(tf.global_variables_initializer())

    # to the placeholder.
    print('Example: tf.get_variable:')
    print(my_variable.eval())

    print('\nExample: tf.get_variable:')
    print(my_int_variable.eval())

    print('\nExample: tf.get_variable:')
    print(other_variable.eval())

    print('\nRank 0:')
    print('String scalar variable: ', mammal.eval())
    print('Int scalar variable: ', ignition.eval())
    print('Floating scalar variable: ', floating.eval())
    print('Complex scalar variable: ', its_complicated.eval())

    print('\nRank 1:')
    print('String 1D variable: ', mystr.eval())
    print('Int 1D variable: ', cool_numbers.eval())
    print('Floating 1D variable: ', first_primes.eval())
    print('Complex 1D variable: ', its_very_complicated.eval())

    print('\nRank 2:')
    print('2D tensor: ')
    print(mymat.eval())
    print('\n2D tensor: ')
    print(myxor.eval())
    print('\n2D tensor: ')
    print(linear_squares.eval())
    print('\n2D tensor: ')
    print(squarish_squares.eval())
    print('\n2D tensor rank is: ', rank_of_squares.eval())

    print('\n0D tensor/scalar => rank 0 tensor: ', sess.run(tensor0D))
    print('1D tensor/vector => rank 1 tensor: ', sess.run(tensor1D))
    print('1D tensor/vector => rank 1 tensor: ', sess.run(tensor1DVector))

    print('\n2D matrix => rank 2 tensor: ')
    print(sess.run(tensor2D))

    print('\n-------------------------------------------------')
    print('tf.constant -------------------------------------')
    print('-------------------------------------------------')

    print('\n2D tf.constant with defined shape: ')
    print(sess.run(constant))

    print('\n2D tf.constant with defined shape: ')
    print(sess.run(tensor2DAgain))

    print('\nRank 3:')
    print(sess.run(tensor3D))

    print('\nzero tensor type int: ')
    print(sess.run(zero_tensor1))

    print('\nzero tensor type float: ')
    print(sess.run(zero_tensor2))

    print('\none tensor: ')
    print(sess.run(one_tensor))

    print('\nfill tensor: ')
    print(sess.run(fill_tensor))

    print('\nrange tensor: ')
    print(sess.run(range_tensor))

    print('\nlinear tensor: ')
    print(sess.run(lin_tensor))

    print('\nrandom normal tensor: ')
    print(sess.run(rnd_normal))

    print('\nrandom uniform tensor: ')
    print(sess.run(rnd_uniform))

    print('\nreshaped tensor: ')
    print(sess.run(vec1))
    print(sess.run(reshaped))

    print('\nreverse tensor on axis=0: ')
    print(sess.run(vec2))
    print(sess.run(reverseAxis0))

    print('\nreverse tensor on axis=1: ')
    print(sess.run(vec2))
    print(sess.run(reverseAxis1))

    print('\nreverse tensor on axis=0, 1: ')
    print(sess.run(vec2))
    print(sess.run(reverseAxis2))

    print('\nStacking; axis=0')
    print(sess.run(stack1))

    print('\nStacking; axis=1')
    print(sess.run(stack2))

    print('\nString join: ', sess.run(msg))

    # tf.Variable
    print('\n-------------------------------------------------')
    print('tf.Placeholder ----------------------------------')
    print('-------------------------------------------------')

    print('\nFeeding a value into a placeholder tensor:')
    print(t.eval(feed_dict={p: [2.2, 7.3]}))  # Feed a value into the placeholder in order to evaluate it

    print('\n-------------------------------------------------')
    print('tf.Collection -----------------------------------')
    print('-------------------------------------------------')

    print('\nTensor collection: ')
    print(tf.get_collection("my_collection_name"))