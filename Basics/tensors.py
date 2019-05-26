# Carla de Beer
# Created: May 2019
# The basics on tensors (tf.Variable, tf.constant, tf.placeholder)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def create_variables():
    # tf.Variable: represents a shared, persistent state
    my_variable = tf.get_variable('my_variable', [1, 2, 3])
    my_int_variable = tf.get_variable('my_int_variable', [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)
    other_variable = tf.get_variable('other_variable', dtype=tf.int32, initializer=tf.constant([23, 42]))
    return my_variable, my_int_variable, other_variable


def create_tensors_various_rank():
    # Rank 0
    mammal = tf.Variable('Elephant', tf.string)
    ignition = tf.Variable(451, tf.int16)
    floating = tf.Variable(3.14159265359, tf.float64)
    its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

    # Rank 1
    mystr = tf.Variable(['Hello'], tf.string)
    cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
    first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
    its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

    # Rank 2
    mymat = tf.Variable([[7], [11]], tf.int32)
    myxor = tf.Variable([[False, True], [True, False]], tf.bool)
    linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
    squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
    rank_of_squares = tf.rank(squarish_squares)
    return mammal, ignition, floating, its_complicated, mystr, cool_numbers, \
           first_primes, its_very_complicated, mymat, myxor, linear_squares, squarish_squares, rank_of_squares


def create_constant_tensors():
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
    return tensor0D, tensor1D, tensor1DVector, constant, tensor2D, tensor2DAgain, tensor3D


def do_special_tensor_functions():
    # Special tensor functions
    zero_tensor1 = tf.zeros([3], dtype=tf.int32)
    zero_tensor2 = tf.zeros([3])
    one_tensor = tf.ones([4, 3])
    fill_tensor = tf.fill([1, 2, 3], 81.0)
    return zero_tensor1, zero_tensor2, one_tensor, fill_tensor


def create_range_tensor():
    # range
    return tf.range(10, 50, delta=5)


def create_linear_space_tensor():
    # linear
    return tf.lin_space(5., 9., 4)


def create_randomized_tensors():
    # random
    rnd_normal = tf.random_normal([10], dtype=tf.float64)
    rnd_uniform = tf.random_uniform([10], minval=-1, maxval=1, dtype=tf.float32)
    return rnd_normal, rnd_uniform


def create_reshaped_tensors():
    # reshaped
    vec1 = tf.constant([1., 2., 3., 4.])
    reshaped = tf.reshape(vec1, [2, 2])
    return vec1, reshaped


def create_reversed_tensors():
    # reversed
    vec2 = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    reverseAxis0 = tf.reverse(vec2, [0])
    reverseAxis1 = tf.reverse(vec2, [1])
    reverseAxis2 = tf.reverse(vec2, [0, 1])
    return vec2, reverseAxis0, reverseAxis1, reverseAxis2


def create_stacked_tensors():
    # stacked
    t4 = tf.constant([1., 2.])
    t5 = tf.constant([3., 4.])
    t6 = tf.constant([5., 6.])
    stack1 = tf.stack([t4, t5, t6], axis=0)
    stack2 = tf.stack([t4, t5, t6], axis=1)
    return stack1, stack2


def join_strings():
    # String join
    return tf.string_join(['Hello', ' ', 'TensorFlow!'])


def create_placeholder_tensors():
    # tf.placeholder
    p = tf.placeholder(tf.float64)
    t = p + [1.2, 4.4]
    return p, t


def create_tensor_collection():
    # Create a tensor collection
    var = tf.get_variable('var', [1, 2, 3])
    const = tf.constant(17, dtype=tf.int32, shape=[4, 3], name='const', verify_shape=False)
    tf.add_to_collection("my_collection_name", var)
    tf.add_to_collection("my_collection_name", const)


# Calling the methods
[my_variable, my_int_variable, other_variable] = create_variables()

[mammal, ignition, floating, its_complicated, mystr, cool_numbers, first_primes, its_very_complicated, mymat, myxor,
 linear_squares, squarish_squares, rank_of_squares] = create_tensors_various_rank()

[tensor0D, tensor1D, tensor1DVector, constant, tensor2D, tensor2DAgain, tensor3D] = create_constant_tensors()

[zero_tensor1, zero_tensor2, one_tensor, fill_tensor] = do_special_tensor_functions()

range_tensor = create_range_tensor()

lin_tensor = create_linear_space_tensor()

[rnd_normal, rnd_uniform] = create_randomized_tensors()

[vec1, reshaped] = create_reshaped_tensors()

[vec2, reverseAxis0, reverseAxis1, reverseAxis2] = create_reversed_tensors()

[stack1, stack2] = create_stacked_tensors()

msg = join_strings()

[p, t] = create_placeholder_tensors()

create_tensor_collection()

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
    np.testing.assert_array_almost_equal(my_int_variable.eval(), np.array([[[0, 0, 0],
                                                                            [0, 0, 0]]], dtype=int))

    print('\nExample: tf.get_variable:')
    print(other_variable.eval())
    np.testing.assert_array_almost_equal(other_variable.eval(), np.array([23, 42], dtype=int))

    print('\nRank 0:')
    print('String scalar variable: ', mammal.eval())
    assert (mammal.eval() == b'Elephant')

    print('Int scalar variable: ', ignition.eval())
    assert (ignition.eval() == 451)

    print('Floating scalar variable: ', floating.eval())
    np.testing.assert_array_almost_equal(floating.eval(), 3.1415927)

    print('Complex scalar variable: ', its_complicated.eval())
    np.testing.assert_array_almost_equal(its_complicated.eval(), (12.3 - 4.85j))

    print('\nRank 1:')
    print('String 1D variable: ', mystr.eval())
    assert (mystr.eval() == [b'Hello'])

    print('Int 1D variable: ', cool_numbers.eval())
    np.testing.assert_array_almost_equal(cool_numbers.eval(), np.array([3.14159, 2.71828], dtype=complex))

    print('Floating 1D variable: ', first_primes.eval())
    np.testing.assert_array_almost_equal(first_primes.eval(), np.array([2, 3, 5, 7, 11], dtype=int))

    print('Complex 1D variable: ', its_very_complicated.eval())
    np.testing.assert_array_almost_equal(its_very_complicated.eval(),
                                         np.array([12.3 - 4.85j, 7.5 - 6.23j], dtype=complex))

    print('\nRank 2:')
    print('2D tensor: ')
    print(mymat.eval())
    np.testing.assert_array_almost_equal(mymat.eval(), np.array([[7], [11]], dtype=int))

    print('\n2D tensor: ')
    print(myxor.eval())
    np.testing.assert_array_almost_equal(myxor.eval(), np.array([[False, True],
                                                                 [True, False]], dtype=bool))

    print('Shape myxor:', myxor.get_shape())

    print('\n2D tensor: ')
    print(linear_squares.eval())
    np.testing.assert_array_almost_equal(linear_squares.eval(), np.array([[4], [9], [16], [25]], dtype=int))

    print('Shape linear_squares:', linear_squares.get_shape())
    print('\n2D tensor: ')
    print(squarish_squares.eval())
    np.testing.assert_array_almost_equal(squarish_squares.eval(), np.array([[4, 9], [16, 25]], dtype=int))

    print('\n2D tensor rank is: ', rank_of_squares.eval())
    assert (rank_of_squares.eval() == 2)

    print('\n0D tensor/scalar => rank 0 tensor: ', sess.run(tensor0D))
    np.testing.assert_array_almost_equal(sess.run(tensor0D), np.array(4, dtype=int))

    print('1D tensor/vector => rank 1 tensor: ', sess.run(tensor1D))
    np.testing.assert_array_almost_equal(sess.run(tensor1D), np.array([4], dtype=int))

    print('1D tensor/vector => rank 1 tensor: ', sess.run(tensor1DVector))
    np.testing.assert_array_almost_equal(sess.run(tensor1DVector), np.array([4, 2], dtype=int))

    print('\n2D matrix => rank 2 tensor: ')
    print(sess.run(tensor2D))
    np.testing.assert_array_almost_equal(sess.run(tensor2D), np.array([[1, 2], [5, 7]], dtype=int))

    print('\n-------------------------------------------------')
    print('tf.constant -------------------------------------')
    print('-------------------------------------------------')

    print('\n2D tf.constant with defined shape: ')
    print(sess.run(constant))
    np.testing.assert_array_almost_equal(sess.run(constant), np.array([[17, 17, 17],
                                                                       [17, 17, 17],
                                                                       [17, 17, 17],
                                                                       [17, 17, 17]], dtype=int))

    print('\n2D tf.constant with defined shape: ')
    print(sess.run(tensor2DAgain))
    np.testing.assert_array_almost_equal(sess.run(tensor2DAgain), np.array([[1, 2], [5, 7], [9, 11]], dtype=int))

    print('\nRank 3:')
    print(sess.run(tensor3D))
    np.testing.assert_array_almost_equal(sess.run(tensor3D), np.array([[[5, 5, 5],
                                                                        [5, 5, 5],
                                                                        [5, 5, 5],
                                                                        [5, 5, 5]],
                                                                       [[5, 5, 5],
                                                                        [5, 5, 5],
                                                                        [5, 5, 5],
                                                                        [5, 5, 5]]], dtype=int))

    print('\nzero tensor type int: ')
    print(sess.run(zero_tensor1))
    np.testing.assert_array_almost_equal(sess.run(zero_tensor1), np.array([0, 0, 0], dtype=int))

    print('\nzero tensor type float: ')
    print(sess.run(zero_tensor2))
    np.testing.assert_array_almost_equal(sess.run(zero_tensor2), np.array([0., 0., 0.], dtype=float))

    print('\none tensor: ')
    print(sess.run(one_tensor))
    np.testing.assert_array_almost_equal(sess.run(one_tensor), np.array([[1., 1., 1.],
                                                                         [1., 1., 1.],
                                                                         [1., 1., 1.],
                                                                         [1., 1., 1.]], dtype=float))

    print('\nfill tensor: ')
    print(sess.run(fill_tensor))
    np.testing.assert_array_almost_equal(sess.run(fill_tensor), np.array([[[81., 81., 81.],
                                                                           [81., 81., 81.]]], dtype=float))

    print('\nrange tensor: ')
    print(sess.run(range_tensor))
    np.testing.assert_array_almost_equal(sess.run(range_tensor), np.array([10, 15, 20, 25, 30, 35, 40, 45], dtype=int))

    print('\nlinear tensor: ')
    print(sess.run(lin_tensor))
    np.testing.assert_array_almost_equal(sess.run(lin_tensor), np.array([5., 6.3333335, 7.666667, 9.], dtype=float))

    print('\nrandom normal tensor: ')
    print(sess.run(rnd_normal))

    print('\nrandom uniform tensor: ')
    print(sess.run(rnd_uniform))

    print('\nreshaped tensor: ')
    print(sess.run(vec1))
    print(sess.run(reshaped))
    np.testing.assert_array_almost_equal(sess.run(reshaped), np.array([[1., 2.],
                                                                       [3., 4.]], dtype=float))

    print('\nreverse tensor on axis=0: ')
    print(sess.run(vec2))
    print(sess.run(reverseAxis0))
    np.testing.assert_array_almost_equal(sess.run(reverseAxis0), np.array([[4., 5., 6.],
                                                                           [1., 2., 3.]], dtype=float))

    print('\nreverse tensor on axis=1: ')
    print(sess.run(vec2))
    print(sess.run(reverseAxis1))
    np.testing.assert_array_almost_equal(sess.run(reverseAxis1), np.array([[3., 2., 1.],
                                                                           [6., 5., 4.]], dtype=float))

    print('\nreverse tensor on axis=0, 1: ')
    print(sess.run(vec2))
    print(sess.run(reverseAxis2))
    np.testing.assert_array_almost_equal(sess.run(reverseAxis2), np.array([[6., 5., 4.],
                                                                           [3., 2., 1.]], dtype=float))

    print('\nStacking; axis=0')
    print(sess.run(stack1))
    np.testing.assert_array_almost_equal(sess.run(stack1), np.array([[1., 2.],
                                                                     [3., 4.],
                                                                     [5., 6.]], dtype=float))

    print('\nStacking; axis=1')
    print(sess.run(stack2))
    np.testing.assert_array_almost_equal(sess.run(stack2), np.array([[1., 3., 5.],
                                                                     [2., 4., 6.]], dtype=float))

    print('\nString join: ', sess.run(msg))
    assert (msg.eval() == b'Hello TensorFlow!')

    # tf.Variable
    print('\n-------------------------------------------------')
    print('tf.Placeholder ----------------------------------')
    print('-------------------------------------------------')

    print('\nFeeding a value into a placeholder tensor:')
    param = feed_dict = {p: [2.2, 7.3]}
    print(t.eval(param))  # Feed a value into the placeholder in order to evaluate it
    np.testing.assert_array_almost_equal(t.eval(param), np.array([3.4, 11.7], dtype=float))

    print('\n-------------------------------------------------')
    print('tf.Collection -----------------------------------')
    print('-------------------------------------------------')

    print('\nTensor collection: ')
    print(tf.get_collection("my_collection_name"))
