import tensorflow as tf


def tensor(x):
    return tf.constant(x)


def sum(x, axis=None):
    return tf.reduce_sum(x, axis=axis)
