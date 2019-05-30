# Carla de Beer
# Created: May 2019
# Basic tensor operations

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

matrix_a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def calculate_diagonal_matrix():
    return tf.diag([4, 5, -6, 3, 1])


def calculate_tensor_dot():
    a = tf.constant([5, 2, 8])
    b = tf.constant([3, 1, 5])
    return tf.tensordot(a, b, axes=0)


def calculate_cross_product():
    a = tf.constant([5, 2, 8])
    b = tf.constant([3, 1, 5])
    return tf.cross(a, b)


def calculate_trace():
    return tf.trace(matrix_a)


def calculate_transpose():
    return tf.transpose(matrix_a)


def calculate_eye():
    return tf.eye(3, 4, batch_shape=None, dtype=tf.float32, name='identity')


def calculate_mat_mul():
    a = tf.constant([[2, 3, 5], [4, 2, 6]])
    b = tf.constant([[2, 3], [2, -2], [2, 1]])
    return tf.matmul(a, b)


def calculate_norm():
    batch = tf.constant(np.random.rand(3, 2, 3, 6))
    return tf.norm(batch, axis=3)


def calculate_matrix_solve():
    m = tf.constant([[2, -2, 1], [1, 3, -2], [3, -1, -1]], dtype=tf.float32)
    x = tf.constant([[3, 1, 1], [4, 1, 2], [5, 2, 1]], dtype=tf.float32)
    return tf.matrix_solve(m, x)


def do_einsum():
    m1 = tf.constant([[1, 2], [3, 4]])
    return tf.einsum('ij->ji', m1)


diagonal = calculate_diagonal_matrix()

tensor_dot = calculate_tensor_dot()

cross = calculate_cross_product()

trace = calculate_trace()

transpose = calculate_transpose()

eye = calculate_eye()

mat_mul = calculate_mat_mul()

norm = calculate_norm()

mat_solve = calculate_matrix_solve()

einsum = do_einsum()

# Execute the operations
with tf.Session() as sess:
    print('\nDiagonal tensor:')
    print(sess.run(diagonal))
    np.testing.assert_array_almost_equal(sess.run(diagonal), np.array([[4, 0, 0, 0, 0],
                                                                       [0, 5, 0, 0, 0],
                                                                       [0, 0, - 6, 0, 0],
                                                                       [0, 0, 0, 3, 0],
                                                                       [0, 0, 0, 0, 1]], dtype=float))

    print('\nTensordot tensor:')
    print(sess.run(tensor_dot))
    np.testing.assert_array_almost_equal(sess.run(tensor_dot), np.array([[15, 5, 25],
                                                                         [6, 2, 10],
                                                                         [24, 8, 40]], dtype=float))

    print('\nCross product tensor:')
    print(sess.run(cross))
    np.testing.assert_array_almost_equal(sess.run(cross), np.array([2, -1, -1], dtype=float))

    print('\nTrace tensor returns sum of diagonals:')
    print(sess.run(trace))
    np.testing.assert_array_almost_equal(sess.run(trace), 15)

    print('\nTranspose tensor:')
    print(sess.run(transpose))
    np.testing.assert_array_almost_equal(sess.run(transpose), np.array([[1, 4, 7],
                                                                        [2, 5, 8],
                                                                        [3, 6, 9]], dtype=float))

    print('\nIdentity tensor:')
    print(sess.run(eye))
    np.testing.assert_array_almost_equal(sess.run(eye), np.array([[1., 0., 0., 0.],
                                                                  [0., 1., 0., 0.],
                                                                  [0., 0., 1., 0.]], dtype=float))

    print('\nMatrix multiplication:')
    print(sess.run(mat_mul))
    np.testing.assert_array_almost_equal(sess.run(mat_mul), np.array([[20, 5],
                                                                      [24, 14]], dtype=float))

    print('\nMatrix normalization:')
    print(sess.run(norm))

    print('\nMatrix solve:')
    print(sess.run(mat_solve))
    np.testing.assert_array_almost_equal(sess.run(mat_solve), np.array([[2.1999998e+00, 6.0000002e-01, 9.9999994e-01],
                                                                        [9.9999982e-01, - 1.7881394e-08, 9.9999994e-01],
                                                                        [5.9999967e-01, - 2.0000003e-01,
                                                                         9.9999982e-01]],
                                                                       dtype=float))

    print('\neinsum:')
    print(sess.run(einsum))
    np.testing.assert_array_almost_equal(sess.run(einsum), np.array([[1, 3],
                                                                     [2, 4]],
                                                                    dtype=float))
