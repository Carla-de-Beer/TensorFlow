# Carla de Beer
# Created: May 2019
# Basic tensor mathematical operations

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

values1 = [10, 0, 30, 40, 50, 60, 3, 1, 2]
values2 = [[1, 2], [3, 4]]
values3 = [[-4.5, 3.2], [6.7, 2.1]]

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 0], [0, 6]], dtype=tf.float32)
d = tf.constant([[52, 0, 0], [0, 2, 88], [0, 34, 67]])

f = tf.constant(values2, dtype=tf.float32)
g = tf.constant(values2, dtype=tf.float32)
h = tf.constant([[2, 2], [4, 4]], dtype=tf.float32)

max_tensor = tf.constant([[5, 220, 55], [6, 2, 88]])

mat_a = tf.constant([[2, 3], [1, 2], [4, 5]], dtype=tf.int32)
mat_b = tf.constant([[6, 4, 1], [3, 7, 2]], dtype=tf.int32)


def calculate_maths_constant_tensors():
    const_a = tf.constant(3.6)
    const_b = tf.constant(1.2)
    total = const_a + const_b
    quot = tf.divide(const_a, const_b)
    return total, quot


def calculate_vector_multiplication():
    vec_a = tf.linspace(0.0, 3.0, 4)
    vec_b = tf.fill([4, 1], 2.0)
    prod = tf.multiply(vec_a, vec_b)
    dot = tf.tensordot(vec_a, vec_b, 1)
    return prod, dot


def calculate_matrix_multiplication():
    return tf.matmul(mat_a, mat_b)


def calculate_cumulative_product():
    cum_prod0 = tf.math.cumprod([a, b], axis=0)
    cum_prod1 = tf.math.cumprod([a, b], axis=1)
    return cum_prod0, cum_prod1


def calculate_division():
    return tf.math.divide(b, a)


def calculate_accumulation():
    return tf.math.accumulate_n([a, b, b])


def calculate_subtraction():
    rand_a = tf.random_normal([3], 2.0)
    rand_b = tf.random_uniform([3], 1.0, 4.0)
    diff1 = tf.subtract(rand_a, rand_b)

    fix_a = tf.constant([[[1., 2.],
                          [7., 8.]],
                         [[5., 0.],
                          [0., 0.]]], dtype=tf.float32)
    fix_b = tf.constant([[[-6., 4.],
                          [0., 8.]],
                         [[1., 7.],
                          [1., -5.]]], dtype=tf.float32)
    diff2 = tf.subtract(fix_a, fix_b)
    return diff1, diff2


def calculate_floor_ceil():
    floor = tf.math.floor(values3)
    ceil = tf.math.ceil(values3)
    return floor, ceil


def calculate_less():
    less1 = tf.math.less(a, b)
    less2 = tf.math.less(b, a)
    return less1, less2


def calculate_non_zero():
    return tf.math.count_nonzero(d)


def calculate_tensor_equality():
    equal1 = tf.math.equal(f, b)
    equal2 = tf.math.equal(g, h)
    equal3 = tf.math.equal(f, g)
    return equal1, equal2, equal3


def calculate_arg_min_max():
    max1 = tf.math.argmax(values1, 0)
    max2 = tf.math.argmax(mat_a, axis=0)
    min1 = tf.math.argmin(values1, 0)
    min2 = tf.math.argmin(mat_a)

    x_max0 = tf.argmax(max_tensor, axis=0)
    x_max1 = tf.argmax(max_tensor, axis=1)
    return max1, max2, min1, min2, x_max0, x_max1


def calculate_exp():
    return tf.math.exp(g)


def calculate_error_function():
    y = tf.constant(values1, dtype=tf.float32)
    return tf.erf(y)


def calculate_trig_calculations():
    theta = tf.constant([degree2radians(30), degree2radians(45)], dtype=tf.float32, name=None)
    sin = tf.math.sin(theta)
    cos = tf.math.cos(theta)
    return sin, cos


# Convert from degrees to radians
def degree2radians(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


[total, quot] = calculate_maths_constant_tensors()

[prod, dot] = calculate_vector_multiplication()

mat_prod = calculate_matrix_multiplication()

[cum_prod0, cum_prod1] = calculate_cumulative_product()

division = calculate_division()

accumulation = calculate_accumulation()

[diff1, diff2] = calculate_subtraction()

[floor, ceil] = calculate_floor_ceil()

[less1, less2] = calculate_less()

non_zero = calculate_non_zero()

[equal1, equal2, equal3] = calculate_tensor_equality()

[max1, max2, min1, min2, x_max0, x_max1] = calculate_arg_min_max()

exp = calculate_exp()

erf = calculate_error_function()

[sin, cos] = calculate_trig_calculations()

t1 = tf.constant(1.2)
t2 = tf.constant(3.5)

intSession = tf.InteractiveSession()
print('\nInteractive Session: ')
print('\nProduct: ', tf.multiply(t1, t2).eval())

graph = tf.get_default_graph()

# Execute the operations
with tf.Session() as sess:
    print('\nSum: %f' % sess.run(total))
    np.testing.assert_array_almost_equal(sess.run(total), 4.800000)

    print('\nQuotient: %f' % sess.run(quot))
    np.testing.assert_array_almost_equal(sess.run(quot), 3.000000)

    print('\nElement-wise product: ')
    print(sess.run(prod))
    np.testing.assert_array_almost_equal(sess.run(prod), np.array([[0., 2., 4., 6.],
                                                                   [0., 2., 4., 6.],
                                                                   [0., 2., 4., 6.],
                                                                   [0., 2., 4., 6.]], dtype=float))

    print('\nDot product: ', sess.run(dot))
    np.testing.assert_array_almost_equal(sess.run(dot), np.array([12.], dtype=float))

    print('\nMatrix product: ')
    print(sess.run(mat_prod))
    assert np.all(sess.run(mat_prod) == np.array([[21, 29, 8],
                                                  [12, 18, 5],
                                                  [39, 51, 14]]))

    print('\nCumulative product axis=0', sess.run(cum_prod0))
    np.testing.assert_array_almost_equal(sess.run(cum_prod0), np.array([[[1., 2.],
                                                                         [3., 4.]],
                                                                        [[5., 0.],
                                                                         [0., 24.]]], dtype=float))

    print('\nCumulative product axis=1', sess.run(cum_prod1))
    np.testing.assert_array_almost_equal(sess.run(cum_prod1), np.array([[[1., 2.],
                                                                         [3., 8.]],
                                                                        [[5., 0.],
                                                                         [0., 0.]]], dtype=float))

    print('\nDivide', sess.run(division))
    np.testing.assert_array_almost_equal(sess.run(division), np.array([[5., 0.],
                                                                       [0., 1.5]], dtype=float))

    print('\nAccumulate n', sess.run(accumulation))
    np.testing.assert_array_almost_equal(sess.run(accumulation), np.array([[11., 2.],
                                                                           [3., 16.]], dtype=float))

    print('\nDifference (random): ', sess.run(diff1))
    print('\nDifference (fixed): ', sess.run(diff2))
    np.testing.assert_array_almost_equal(sess.run(diff2), np.array([[[7., - 2.],
                                                                     [7., 0.]],
                                                                    [[4., - 7.],
                                                                     [-1., 5.]]], dtype=float))

    print('\nFloor', sess.run(floor))
    np.testing.assert_array_almost_equal(sess.run(floor), np.array([[-5., 3.],
                                                                    [6., 2.]], dtype=float))

    print('\nCeil', sess.run(ceil))
    np.testing.assert_array_almost_equal(sess.run(ceil), np.array([[-4., 4.],
                                                                   [7., 3.]], dtype=float))

    print('\nLess', sess.run(less1))
    assert np.all(sess.run(less1) == np.array([[True, False], [False, True]], dtype=bool))

    print('\nLess', sess.run(less2))
    assert np.all(sess.run(less2) == np.array([[False, True], [True, False]], dtype=bool))

    print('\nCount non-zero', sess.run(non_zero))
    assert (sess.run(non_zero) == 5)

    print('\nEqual (false)', sess.run(equal1))
    assert np.all(sess.run(equal1) == np.array([[False, False],
                                                [False, False]], dtype=bool))

    print('\nEqual (true/false)', sess.run(equal2))
    assert np.all(sess.run(equal2) == np.array([[False, True],
                                                [False, True]], dtype=bool))

    print('\nEqual (true)', sess.run(equal3))
    assert np.all(sess.run(equal3) == np.array([[True, True],
                                                [True, True]], dtype=bool))

    print('\nArgmax: ', sess.run(max1))
    assert np.all(sess.run(max1) == 5)

    print('\nArgmax: ', sess.run(max2))
    assert np.all(sess.run(max2) == np.array([2, 2]))

    print('\nArgmin: ', sess.run(min1))
    assert np.all(sess.run(min1) == 1)

    print('\nArgmin: ', sess.run(min2))
    assert np.all(sess.run(min2) == np.array([1, 1]))

    # Returns the index values of the largest values along a specific axis
    print('Argmax axis=0', sess.run(x_max0))
    assert np.all(sess.run(x_max0) == np.array([1, 0, 1]))

    print('Argmax axis=1', sess.run(x_max1))
    assert np.all(sess.run(x_max1) == np.array([1, 2]))

    print('\nExp', sess.run(exp))
    np.testing.assert_array_almost_equal(sess.run(exp), np.array([[2.7182817, 7.389056],
                                                                  [20.085537, 54.59815]], dtype=float), decimal=5)

    print('\nError function', sess.run(erf))
    np.testing.assert_array_almost_equal(sess.run(erf), np.array([1., 0., 1., 1., 1., 1., 0.9999779,
                                                                  0.8427008, 0.9953223], dtype=float))

    print('\nsin theta (radians)', sess.run(sin))
    np.testing.assert_array_almost_equal(sess.run(sin), np.array([0.5, 0.70710677], dtype=float))

    print('\ncos theta (radians)', sess.run(cos))
    np.testing.assert_array_almost_equal(sess.run(cos), np.array([0.8660254, 0.70710677], dtype=float))

    print(tf.get_default_graph().get_operations())
